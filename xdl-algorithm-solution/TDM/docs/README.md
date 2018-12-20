Tree-based Deep Match（TDM）是由阿里妈妈精准定向广告算法团队自主研发的基于深度学习上的大规模（千万级+）推荐系统算法框架。在大规模推荐系统的实践中，基于商品的协同过滤算法（Item-CF）是应用较为广泛的，而受到图像检索的启发，基于内积模型的向量检索算法也崭露头角，这些推荐算法产生了一定的效果，但因为受限于算法模型本身的理论限制，推荐的最终结果并不十分理想。近些年，深度学习技术逐渐兴起，在包括如图像、自然语言处理、语音等领域的应用产生了巨大的效果推动。受制于候选集合的大规模，在推荐系统里全面应用深度学习进行计算存在效果低下的问题。针对这一难题TDM原创性的提出了以树结构组织大规模候选，建立目标（兴趣）的层次化依赖关系，并通过逐层树检索的方式进行用户对目标（兴趣）的偏好计算，从而实现用户目标（兴趣）的最终推荐。无论是在公开数据集上离线测试结果[1]，还是在阿里妈妈实际业务的在线测试上，TDM都取得了非常显著的效果提升。

# 基础知识

## 写在最前面

本文主要面向推荐系统的学术爱好者和实际从业者介绍TDM系统的组成原理，TDM开源项目的使用指导，以及如何将TDM算法应用到使用者实际业务中去。下述介绍假定读者具备一定的推荐系统/深度学习的概念了解，基本的数据结构与算法理解，以及基本编程语言（C++，Python）的掌握，如果读者对这些技术的了解存在疑虑，可参考下述知识点进行学习。

## 需要的前提知识

- 基本概念
  - [推荐系统](https://en.wikipedia.org/wiki/Recommender_system)：推荐技术提供用户对某个目标（兴趣）的偏好程度预测能力，基于偏好预测排序输出用户喜好的目标（兴趣）集合。
- [深度学习](https://en.wikipedia.org/wiki/Deep_learning)：基于深度神经网络结构的机器学习算法分支。
- 数据结构与算法
  - [树](https://en.wikipedia.org/wiki/Tree_)：树是计算机科学中数据结构的经典类型，在数据组织上具有良好的效率。
- [BeamSearch算法](https://en.wikipedia.org/wiki/Beam_search)：一种启发式的贪心搜索方法。
- [Kmeans聚类算法](https://en.wikipedia.org/wiki/K-means_clustering)：一种基于向量量化的无监督聚类算法。
- 语言基础
  - [C++语言](https://en.wikipedia.org/wiki/C++)：通用编程语言。
  - [Python语言](https://en.wikipedia.org/wiki/Python)：通用脚本编程语言。

## 文档内容说明

本文旨在介绍TDM以及如何使用TDM开源项目进行实际业务生产所用，阅读完成后，你可以了解到：

- TDM的基本系统组成
- TDM开源代码的运行和使用
- 应用TDM到具体实践的方法

受限于篇幅以及主旨，以下内容本文不涉及，或请参阅相关文档：

- [XDL训练平台]()的系统组成和使用。
- 公开数据集([UserBehavior](https://tianchi.aliyun.com/datalab/dataSet.html?dataId=649))的下载、使用和授权。

## TDM适用的范围

TDM面向解决的是大规模推荐问题，自主提出的以树方式组织大规模候选建立层次关系进而支撑层次检索的框架，具有普遍的适用性和优秀的实验效果。TDM算法具有极高的业务适用性，在包括如视频推荐、商品推荐、新闻推荐等业务领域已经成功应用，并取得了非常可观的业务效果。更多的业务领域TDM应用实践正在开展中，本文后续会不断更新TDM的业务实践效果。

# TDM框架介绍

## 算法原理

TDM是为大规模推荐系统设计的、能承载任意先进模型来高效检索用户兴趣的推荐算法解决方案。TDM基于树结构，提出了一套对用户兴趣度量进行层次化建模与检索的方法论，使得系统能直接利高级深度学习模型在全库范围内检索用户兴趣。详细的算法介绍，请参见TDM在KDD会议上的论文[1]，其基本原理是使用树结构对全库item进行索引，然后训练深度模型以支持树上的逐层检索，从而将大规模推荐中全库检索的复杂度由O(n)（n为所有item的量级）下降至O(log n)。

## 系统组成

<div align=center>
<img width="700" src="https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution/TDM/docs/image.png" alt="系统组成" />
</div>

## 树结构

树在TDM框架中承担的是索引结构的角色，即全库item都会通过树结构被索引起来。关于树结构的示例如下：

<div align=center>
<img width="400" src="https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution/TDM/docs/tree_structure.png" alt="树结构示例" />
</div>

如上图所示，树中的每一个叶节点对应库中的一个item；非叶节点表示item的集合。这样的一种层次化结构，体现了粒度从粗到细的item架构。此时，推荐任务转换成了如何从树中检索一系列叶节点，作为用户最感兴趣的item返回。值得一提的是，虽然图中展示的树结构是一个二叉树，但在实际应用中并无此限制。

## 基于树的检索算法

在一些传统的树状索引如二叉搜索树、B-树等结构中，检索过程往往是使用键值在树上进行逐层往下的top 1检索，最终找到一个满足条件的叶节点并返回。而在TDM框架中，基于树结构进行大规模推荐的方法，是每一个用户寻找K个最感兴趣的叶节点。因此，检索过程也做了相应的改变：在逐层往下的检索过程中，每一层都保留K个节点并往下扩展，即经典的BeamSearch方法。这一检索方案兼具效率与精度，剩下的问题是如何得到每层内精准的兴趣判别器，来找到每层内的K个最感兴趣的节点。

## 深度网络模型

深度网络在TDM框架中承担的是用户兴趣判别器的角色，其输出的（用户，节点）对的兴趣度，将被用于检索过程作为寻找每层Top K的评判指标。由于TDM框架具备高效的剪枝能力，因此其能兼容任意先进的深度网络模型来进行全库检索。下图给出了一个深度网络模型的示例：

<div align=center>
<img width="900" src="https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution/TDM/docs/model.png" alt="模型示例" />
</div>

上述网络结构中，在用户特征方面仅使用了用户历史行为，并对历史行为根据其发生时间，进行了时间窗口划分。在节点特征方面，使用的是节点经过embedding后的向量作为输入。此外，模型借助attention结构[2]，将用户行为中和本次判别相关的那部分筛选出来，以实现更精准的判别。

## 网络与树结构的联合训练

树索引结构在TDM框架中起到了两方面的作用，一是在训练过程提供了上溯正采样样本和平层负采样样本；二是在检索过程中决定了选择与剪枝方案。因此，良好的树结构应该能为训练提供易于区分的上层样本，并尽量体现真实的用户兴趣层级结构来帮助检索。基于数据驱动的索引学习及检索模型、索引结构联合训练的出发点，TDM框架进行了一种尝试：使用学习得到的叶节点（即item）embedding向量进行层次化聚类，来生成新的树索引结构。联合训练过程如下图所示：

<div align=center>
<img width="400" src="https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution/TDM/docs/joint_training.png" alt="联合训练示例" />
</div>

检索模型训练、树索引结构学习过程迭代进行，最终得到稳定的结构与模型。

###  

## 评测指标

TDM主要采用召回率、准确率两种评价指标：

- 召回率：R = size({预测集合} ∩ {真集}) / size({真集})
- 准确率：P = size({预测集合} ∩ {真集}) / size({预测集合})



# TDM训练示例

## 单机试验小数据集

用户依照以下步骤可以在随机dummy小数据集上快速验证流程，此dummy数据集为随机生成，仅作快速验证流程使用。如果需要拿到效果，请试验一个具有实际意义的数据集（例如UserBehavior数据集）

- 准备工作目录

```bash
WORKPATH='/your/path/work/tdm_mock'    # 自选一个工作目录
git clone --recursive XDL-Algorithm-Solution.git
# git submodule update --init --recursive     # 如果clone时没有--recursive，那么需要执行本行命令
cp -r XDL-Algorithm-Solution/TDM/script/tdm_ub_att_ubuntu/  "$WORKPATH"
```

- 进入docker

```bash
DOCKER_PATH='your/docker/path/xdl:ubuntu-gpu-mxnet1.3'    # 选择docker镜像地址
sudo docker pull "$DOCKER_PATH"
# 下列命令启动一个名为tdm-mock的容器，当然这个名字可以自行指定，但不能与本机上已有的容器重名
sudo docker run -it --net=host --volume $HOME:$HOME -w $HOME `curl -s http://localhost:3476/docker/cli` --name tdm-mock "$DOCKER_PATH"
# sudo docker exec -ti tdm-mock bash    # 再次进入docker使用该命令，注意tdm-mock应与上述启动容器命令中指定的名字相同
source /etc/profile    # 设置hadoop，以及将当前路径添加到PATH
```

- 编译安装TDM

```bash
apt-get install swig    # 如果已经安装则无需安装
cd  XDL-Algorithm-Solution/TDM/src
mkdir build
cd build
cmake ..
make
cp -r ../python/store/store/ "$WORKPATH"
cp -r ../python/dist_tree/dist_tree/ "$WORKPATH"
cp -r ../python/cluster/ "$WORKPATH"
cp tdm/lib*.so "$WORKPATH"
```

- 分隔数据集，生成原始样本csv

```bash
cd "$WORKPATH/cluster"
python data_cutter.py     \
  --input mock.dat        \    # 输入数据文件名称
  --train mock_train.csv  \    # 输出训练集数据文件名称
  --test mock_test.csv    \    # 输出测试集数据文件名称
  --number 10                  # 测试集用户数
CSV_HDFSPATH='hdfs://your/rawdata/hdfs/path'    # 指定csv产出目录
hadoop fs -ls "$CSV_HDFSPATH"    # 确认该目录之前不存在，若存在则删掉或换个目录
hadoop fs -mkdir "$CSV_HDFSPATH"
hadoop fs -put mock_train.csv "$CSV_HDFSPATH"
hadoop fs -put mock_test.csv "$CSV_HDFSPATH"
```

- 样本产出及树初始化

```bash
cd "$WORKPATH"
UPLOAD_HDFSPATH='hdfs://your/dist_tree/upload/hdfs/path/tree_data'    # 指定样本及树的产出目录
hadoop fs -ls "$UPLOAD_HDFSPATH"    # 确认该目录之前不存在，若存在则删掉或换个目录
hadoop fs -mkdir "$UPLOAD_HDFSPATH"
EMB_HDFSPATH='hdfs://your/emb_converted/hdfs/path'    # 指定训练的embedding结果产出目录
hadoop fs -ls "$EMB_HDFSPATH"       # 确认该目录之前不存在，若存在则删掉或换个目录
hadoop fs -mkdir "$EMB_HDFSPATH"
vim data/tdm.json   # train_rawdata_url 修改为 $CSV_HDFSPATH/mock_train.csv 的完整hdfs路径
                    # test_rawdata_url 修改为 $CSV_HDFSPATH/mock_test.csv 的完整hdfs路径
                    # upload_url 修改为 $UPLOAD_HDFSPATH 的完整hdfs路径，此为树及样本生成目录的hdfs路径
                    # model_url 修改为 $EMB_HDFSPATH 的完整hdfs路径，此为训练模型的sparse参数导出文件的hdfs路径
INIT_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_mock_init'    # 指定样本产出及树初始化的ckpt目录
hadoop fs -ls "$INIT_CKPT_HDFS_PATH"
hadoop fs -mkdir "$INIT_CKPT_HDFS_PATH"    # 如果目录不存在，则创建
hadoop fs -rm -r "$INIT_CKPT_HDFS_PATH/checkpoint"    # 否则删除其中的checkpoint目录
vim config.tree_init.json   # dependent_dirs 修改为 $WORKPATH 的完整路径
                            # docker_image 修改为 $DOCKER_PATH
                            # checkpoint.output_dir 修改为 $INIT_CKPT_HDFS_PATH/checkpoint
python tree_init.py
hadoop fs -ls "$UPLOAD_HDFSPATH/data"
```

- 训练

```bash
cd "$WORKPATH"
TRAIN_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_mock_train'    # 指定训练的ckpt目录
hadoop fs -ls "$TRAIN_CKPT_HDFS_PATH"
hadoop fs -mkdir "$TRAIN_CKPT_HDFS_PATH"    # 如果目录不存在，则创建
hadoop fs -rm -r "$TRAIN_CKPT_HDFS_PATH/checkpoint"    # 否则删除其中的checkpoint目录
vim config.train.json   # dependent_dirs 修改为 $WORKPATH 的完整路径
                        # docker_image 修改为 $DOCKER_PATH
                        # checkpoint.output_dir 修改为 $TRAIN_CKPT_HDFS_PATH/checkpoint
vim data/tdm.json   # 因为是小数据集，因此参数做些相应调整如下：
                    # parall 修改为 4
                    # train_batch_size 修改为 30000
                    # save_checkpoint_interval 修改为 10
                    # predict_batch_size 修改为 100
                    # tdmop_layer_counts 修改为 0,1,2,3,4,5,6,7,8,9,1,1,1,1,1,1,1,1,2,2,3,7,2
vim train.py    # 修改train的代码中DataIO的参数 namenode="hdfs://your/namenode/hdfs/path:9000"，这是样本读取目录的hdfs根结点路径
                # 最下面修改为 is_training=True
python train.py --run_mode=local --config=config.train.json
        # 可能会提示checkpoints找不到，这是正常的寻找ckpt的流程
        # 正常退出则会打印 finish put item_emb
hadoop fs -ls "$TRAIN_CKPT_HDFS_PATH/checkpoint"    # 查看训练保存的ckpt
hadoop fs -ls "$EMB_HDFSPATH"    # 查看生成的 item_emb 文件，大小不应为0
```

- 聚类重新生成树

```bash
cd "$WORKPATH"
CLUSTER_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_mock_cluster'    # 指定聚类的ckpt目录
hadoop fs -ls "$CLUSTER_CKPT_HDFS_PATH"
hadoop fs -mkdir "$CLUSTER_CKPT_HDFS_PATH"    # 如果目录不存在，则创建
hadoop fs -rm -r "$CLUSTER_CKPT_HDFS_PATH/checkpoint"    # 否则删除其中的checkpoint目录
vim config.tree_cluster.json    # dependent_dirs 修改为 $WORKPATH 的完整路径
                                # docker_image 修改为 $DOCKER_PATH
                                # checkpoint.output_dir 修改为 $CLUSTER_CKPT_HDFS_PATH/checkpoint
python tree_cluster.py
hadoop fs -ls "$UPLOAD_HDFSPATH/data" 
```

- 再次训练

同上

- 离线评测

在任意一次训练产出ckpt后即可按需进行评测。

```bash
cd "$WORKPATH"
TRAIN_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_mock_train'    # 指定训练的ckpt目录用于离线评测
hadoop fs -ls "$TRAIN_CKPT_HDFS_PATH"
vim config.predict.json     # dependent_dirs 修改为 $WORKPATH 的完整路径
                            # docker_image 修改为 $DOCKER_PATH
                            # checkpoint.output_dir 修改为 $TRAIN_CKPT_HDFS_PATH/checkpoint
vim train.py    # 修改predict的代码中DataIO的参数 namenode="hdfs://your/namenode/hdfs/path:9000"，注意和上面train不是同一处
                # 最下面修改为 is_training=False
                # 因为是小数据集，因此参数做些相应调整如下：
                # key_value["pr_test_each_layer_retrieve_num"] = "40"
                # key_value["pr_test_final_layer_retrieve_num"] = "20"
python train.py --run_mode=local --config=config.predict.json
```

## 分布式试验小数据集

- 准备任务调度目录

可以直接使用上面单机试验小数据集的工作目录，额外需要做的修改都在下面给出。

```bash
WORKPATH='/your/path/work/tdm_mock'
DISTPATH='/your/path/dist/tdm_mock'
# 将单机运行docker中的工作目录 $WORKPATH 拷贝至分布式任务调度机器上的目录 $DISTPATH ，用于提任务。
cd "$DISTPATH/data"
# 目录中仅保留三个文件 tdm.json、userbehavoir_fc.json、userbehavoir_stat.dat，其余文件删除
# 这些其余文件是单机运行时产出的，虽然不会导致运行结果出错，但提任务时因为要上传整个目录，若不删除则会拖慢上传速度
```

- 分隔数据集，生成原始样本csv

该步骤如上单机产出即可，分布式实验时直接使用。

- 样本产出及树初始化

```bash
cd "$DISTPATH"
INIT_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_mock_init'    # 指定样本产出及树初始化的ckpt目录
hadoop fs -ls "$INIT_CKPT_HDFS_PATH"
hadoop fs -mkdir "$INIT_CKPT_HDFS_PATH"    # 如果目录不存在，则创建
hadoop fs -rm -r "$INIT_CKPT_HDFS_PATH/checkpoint"    # 否则删除其中的checkpoint目录
vim config.tree_init.json   # dependent_dirs 修改为 $DISTPATH 的完整路径
                            # checkpoint.output_dir 修改为 $INIT_CKPT_HDFS_PATH/checkpoint
xdl_submit.py --config config.tree_init.json
```

- 训练

```bash
cd "$DISTPATH"
TRAIN_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_mock_train'    # 指定训练的ckpt目录
hadoop fs -ls "$TRAIN_CKPT_HDFS_PATH"
hadoop fs -mkdir "$TRAIN_CKPT_HDFS_PATH"    # 如果目录不存在，则创建
hadoop fs -rm -r "$TRAIN_CKPT_HDFS_PATH/checkpoint"    # 否则删除其中的checkpoint目录
vim data/tdm.json   # 因为是小数据集，因此参数做些相应调整如下：
                    # train_batch_size 修改为 3000
                    # save_checkpoint_interval 修改为 100
vim config.train.json   # dependent_dirs 修改为 $DISTPATH 的完整路径
                        # checkpoint.output_dir 修改为 $TRAIN_CKPT_HDFS_PATH/checkpoint
                        # 因为是小数据集，因此计算资源申请参数做相应调整如下：
                        #   "worker": {
                        #     "instance_num": 2,
                        #     "cpu_cores": 8,
                        #     "gpu_cores": 1,
                        #     "memory_m": 20000
                        #   },
                        #   "ps": {
                        #     "instance_num": 1,
                        #     "cpu_cores": 8,
                        #     "gpu_cores": 0,
                        #     "memory_m": 16000
                        #   },
vim train.sh    # 最下面修改为 is_training=True
xdl_submit.py --config config.train.json
```

- 聚类重新生成树

```bash
cd "$DISTPATH"
CLUSTER_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_mock_cluster'    # 指定聚类的ckpt目录
hadoop fs -ls "$CLUSTER_CKPT_HDFS_PATH"
hadoop fs -mkdir "$CLUSTER_CKPT_HDFS_PATH"    # 如果目录不存在，则创建
hadoop fs -rm -r "$CLUSTER_CKPT_HDFS_PATH/checkpoint"    # 否则删除其中的checkpoint目录
vim config.tree_cluster.json    # dependent_dirs 修改为 $DISTPATH 的完整路径
                                # checkpoint.output_dir 修改为 $CLUSTER_CKPT_HDFS_PATH/checkpoint
xdl_submit.py --config config.tree_cluster.json
```

- 再次训练

同上

- 离线评测

在任意一次训练产出ckpt后即可按需进行评测。

```bash
cd "$DISTPATH"
TRAIN_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_mock_train'    # 指定训练的ckpt目录用于离线评测
vim config.predict.json     # dependent_dirs 修改为 $DISTPATH 的完整路径
                            # checkpoint.output_dir 修改为 $TRAIN_CKPT_HDFS_PATH/checkpoint
                            # checkpoint.output_dir 修改为 $TRAIN_CKPT_HDFS_PATH/checkpoint
                            # 因为是小数据集，因此计算资源申请参数做相应调整如下：
                            #   "worker": {
                            #     "instance_num": 2,
                            #     "cpu_cores": 8,
                            #     "gpu_cores": 1,
                            #     "memory_m": 20000
                            #   },
                            #   "ps": {
                            #     "instance_num": 1,
                            #     "cpu_cores": 8,
                            #     "gpu_cores": 0,
                            #     "memory_m": 16000
                            #   },
vim train.sh    # 最下面修改为 is_training=False
xdl_submit.py --config config.predict.json
```

## 分布式试验UB数据集

- 准备任务调度目录

```bash
DOCKER_PATH='your/docker/path/xdl:ubuntu-gpu-mxnet1.3'    # 选择docker镜像地址
DISTPATH='/your/path/dist/tdm_ub_att_ubuntu'
git clone --recursive XDL-Algorithm-Solution.git
# git submodule update --init --recursive     # 如果clone时没有--recursive，那么需要执行本行命令
cp -r XDL-Algorithm-Solution/TDM/script/tdm_ub_att_ubuntu/  "$DISTPATH"
```

- 分隔数据集，生成原始样本csv

该步骤见上面。

- 样本产出及树初始化

```bash
cd "$DISTPATH"
UPLOAD_HDFSPATH='hdfs://your/dist_tree/upload/hdfs/path/tree_data'    # 指定样本及树的产出目录
hadoop fs -ls "$UPLOAD_HDFSPATH"    # 确认该目录之前不存在，若存在则删掉或换个目录
hadoop fs -mkdir "$UPLOAD_HDFSPATH"
EMB_HDFSPATH='hdfs://your/emb_converted/hdfs/path'    # 指定训练的embedding结果产出目录
hadoop fs -ls "$EMB_HDFSPATH"       # 确认该目录之前不存在，若存在则删掉或换个目录
hadoop fs -mkdir "$EMB_HDFSPATH"
vim data/tdm.json   # train_rawdata_url 修改为 $CSV_HDFSPATH/mock_train.csv 的完整hdfs路径
                    # test_rawdata_url 修改为 $CSV_HDFSPATH/mock_test.csv 的完整hdfs路径
                    # upload_url 修改为 $UPLOAD_HDFSPATH 的完整hdfs路径，此为树及样本生成目录的hdfs路径
                    # model_url 修改为 $EMB_HDFSPATH 的完整hdfs路径，此为训练模型的sparse参数导出文件的hdfs路径
INIT_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_ub_init'    # 指定样本产出及树初始化的ckpt目录
hadoop fs -ls "$INIT_CKPT_HDFS_PATH"
hadoop fs -mkdir "$INIT_CKPT_HDFS_PATH"    # 如果目录不存在，则创建
hadoop fs -rm -r "$INIT_CKPT_HDFS_PATH/checkpoint"    # 否则删除其中的checkpoint目录
vim config.tree_init.json   # dependent_dirs 修改为 $DISTPATH 的完整路径
                            # docker_image 修改为 $DOCKER_PATH
                            # checkpoint.output_dir 修改为 $INIT_CKPT_HDFS_PATH/checkpoint
                            # 计算资源申请参数可根据用户实际情况调整
xdl_submit.py --config config.tree_init.json    # 该步骤耗时约1~3小时，与实际计算资源有关
hadoop fs -ls "$UPLOAD_HDFSPATH/data" 
```

- 训练

```bash
cd "$DISTPATH"
TRAIN_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_ub_train'    # 指定训练的ckpt目录
hadoop fs -ls "$TRAIN_CKPT_HDFS_PATH"
hadoop fs -mkdir "$TRAIN_CKPT_HDFS_PATH"    # 如果目录不存在，则创建
hadoop fs -rm -r "$TRAIN_CKPT_HDFS_PATH/checkpoint"    # 否则删除其中的checkpoint目录
vim data/tdm.json       # train_epochs 设置为样本需要训练的轮数
vim config.train.json   # dependent_dirs 修改为 $DISTPATH 的完整路径
                        # docker_image 修改为 $DOCKER_PATH
                        # checkpoint.output_dir 修改为 $TRAIN_CKPT_HDFS_PATH/checkpoint
                        # 计算资源申请参数可根据用户实际情况调整
vim train.py    # 修改train的代码中DataIO的参数 namenode="hdfs://your/namenode/hdfs/path:9000"，这是样本读取目录的hdfs根结点路径
                # 最下面修改为 is_training=True
xdl_submit.py --config config.train.json    # 该步骤耗时约10小时或更久，与实际计算资源及样本轮数有关
hadoop fs -ls "$TRAIN_CKPT_HDFS_PATH/checkpoint"    # 查看训练保存的ckpt
hadoop fs -ls "$EMB_HDFSPATH"    # 查看生成的 item_emb 文件，大小不应为0
```

- 聚类重新生成树

```bash
cd "$DISTPATH"
CLUSTER_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_ub_cluster'    # 指定聚类的ckpt目录
hadoop fs -ls "$CLUSTER_CKPT_HDFS_PATH"
hadoop fs -mkdir "$CLUSTER_CKPT_HDFS_PATH"    # 如果目录不存在，则创建
hadoop fs -rm -r "$CLUSTER_CKPT_HDFS_PATH/checkpoint"    # 否则删除其中的checkpoint目录
vim config.tree_cluster.json    # dependent_dirs 修改为 $DISTPATH 的完整路径
                                # docker_image 修改为 $DOCKER_PATH
                                # checkpoint.output_dir 修改为 $CLUSTER_CKPT_HDFS_PATH/checkpoint
                                # 计算资源申请参数可根据用户实际情况调整
xdl_submit.py --config config.tree_cluster.json    # 该步骤耗时约1~3小时，与实际计算资源有关
hadoop fs -ls "$UPLOAD_HDFSPATH/data" 
```

- 再次训练

同上

- 离线评测

在任意一次训练产出ckpt后即可按需进行评测。

```bash
cd "$DISTPATH"
TRAIN_CKPT_HDFS_PATH='hdfs://your/hdfs/path/tdm_ub_train'    # 指定训练的ckpt目录用于离线评测
vim config.predict.json     # dependent_dirs 修改为 $DISTPATH 的完整路径
                            # docker_image 修改为 $DOCKER_PATH
                            # checkpoint.output_dir 修改为 $TRAIN_CKPT_HDFS_PATH/checkpoint
                            # checkpoint.output_dir 修改为 $TRAIN_CKPT_HDFS_PATH/checkpoint
                            # 计算资源申请参数可根据用户实际情况调整
vim train.py    # 修改predict的代码中DataIO的参数 namenode="hdfs://your/namenode/hdfs/path:9000"，注意和上面train不是同一处
                # 最下面修改为 is_training=False
xdl_submit.py --config config.predict.json    # 该步骤耗时约5-15分钟，与实际计算资源有关
```


## 数据处理

本小结会简要描述数据处理阶段主要完成的功能以及输入输出格式，主要包括生成训练样本、测试样本、初始树.

### 初始数据准备

在进行数据处理之前, 我们先要准备原始训练数据以及原始测试数据, 这两份数据格式均为以逗号分隔的文本文件, 每行文本为一个行为记录. 比如下面的示例数据为id为1的用户的5条点击行为(以逗号分隔的各字段含义为user_id, item_id, category_id, behavior_type, timestamp).

```csv
1,2268318,2520377,pv,1511544070
1,2333346,2520771,pv,1511561733
```

### 数据处理脚本

- 生成样本格式

生成的样本为文本文件, 每行为一个行为序列, 格式为:

```csv
sample id | group id | features | dense | label | ts
其中features为; 分隔的Key@Value序列
```

下面是行为序列样本示例:

```csv
619706_13|619706_13|train_unit_id@3829251:1.0;item_55@1180190;item_53@2964905;item_54@4871;item_56@2416791;item_57@1420124;item_58@1165085;item_59@33793;item_65@917114;item_64@4080531;item_67@3915603;item_66@511224;item_61@2694865;item_60@5159307;item_63@2638297;item_62@511224;item_69@917114;item_68@1400292;item_52@629303||1.0|
239290_1|239290_1|train_unit_id@4075487:1.0;item_65@4940273;item_64@3654350;item_67@1314642;item_66@1042927;item_69@345076;item_68@2365838||1.0|
273937_1|273937_1|train_unit_id@2137809:1.0;item_65@170544;item_64@3433418;item_67@848255;item_66@2127356;item_69@3699491;item_68@2122609||1.0|
```

- 特征配置文件示例

```json
{
  "features": {
    "item_1": {
      "start": 0,
      "end": 10,
      "value": 1
    },
    "item_2": {
      "start": 10,
      "end": 20,
      "value": 1
    }
}
```

此特征配置文件表示将用户的第0个到第10个行为序列作为一个特征组，命名为item_1;将用户的第10个到第20个行为序列作为一个特征组，命名为item_2;

- 数据处理流程

数据处理分为数据读取, 生成训练样本, 生成概率统计文件, 生成测试样本, 生成初始化树(PB格式), 整个流程可以用下面的流程图描述.

<div align=center>
<img src="https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution/TDM/docs/TDM_Data_Init_Process.png">
</div>

- 扩展及修改

数据处理阶段的主要逻辑都在generator.py脚本中, 脚本按上述流程组织成模块, 在需要时候非常易于扩展及修改, 下面简单描述下各流程的主要逻辑及可能的修改.

```
>> Read And Parse Raw Data
```

该流程读取并解析原始输入的训练或者测试数据, 并返回一个按列组织的字典, 及每个Key是一个列名, 对应的Value是该列所有的数据(实际存储为一个numpy array). 这部分逻辑比较简单, 主要就是按行读取, 并按逗号(可以扩展为其他分隔符)分割, 并将各列Parse成指定的数据类型. 用户实际使用时候, 可以根据自己的数据格式修改相应的读取及解析逻辑,  比如支持其他的辅助列.

```
>> Generate Train Sample
```

这一部分生成用户行为序列, 并写入样本文件中, 样本格式见上文. 这一部分逻辑分为两步: 先将各用户的行为聚合起来, 实际上就是生成一个以用户id为Key, 用户点击Item Id列表为Value的字典, 注意这里会对各用户的点击的Item进行排序, 排序的依据是点击行为发生的时间戳; 完成用户行为聚合后, 再将用户的行为按样本序列长度切割成样本.

```
>>>Generate Stat Info
```

该模块的功能为统计各Item在训练样本中出现的概率, 并将其写入统计文件.

```
Generate Test Sample
```

该模块逻辑基本和Generate Train Sample相同, 略.

```
Init Tree
```

该模块生成初始化树, 实际是一个逐层聚类过程, 将所有的Item按其属性不断的进行二分聚类, 直至每个类只有单个的Item属性, 聚类的最终效果是形成了一颗多层的二叉树, 所有的item均属于叶子节点, 所有的叶子结点从左到右相当于形成了一个有序的序列. 在实际的处理, 我们采用直接按Item的Category进行排序, 并在排序之后不断二分来形成初始化树, 树的格式及存储见下面树构建部分.在用户实际业务中用户可根据自己的业务场景修改相应的排序逻辑以支持不同的树初始化方式.

## 树构建

树构建阶段主要完成聚类树的生成, 该阶段的输入是根据初始树联合训练生成的item的向量, 并对向量进行聚类生成新的聚类树.

### 输入数据及格式

树构建阶段的输入是用初始树联合训练产生的向量, 输入的格式为文本格式, 每一行描述一个Item对应的id和向量, 均为数值类型, 各数值之间以逗号分隔, 下面为截选的部分输入数据.

```csv
2515040,-0.508644402027,-0.016029631719,-0.20682746172,-0.397063672543,-0.00334448833019,-0.960261583328,0.316593915224,-0.636762738228,-0.217385306954,0.0592824667692,0.35680475831,-0.43331822753,-0.369034737349,0.351467847824,0.0969775170088,0.265370905399,0.0815298631787,-0.389724433422,-0.339153647423,0.273165374994,-0.00598054006696,-0.488672584295,0.405939608812,-0.492451280355
235900,-0.0178719386458,0.117409579456,0.0135170938447,0.208914965391,0.270535558462,-0.295207798481,-0.177082359791,-0.312212228775,0.449806898832,0.338447093964,0.0621097162366,0.327057540417,0.126456350088,0.0875944793224,0.577477931976,-0.351881921291,0.138958856463,-0.538168728352,0.329808682203,-0.239835038781,0.19346319139,0.393561393023,0.111480668187,0.317542433739
3148360,0.359621971846,-0.127544790506,-0.297782152891,-0.368366599083,0.223647251725,-0.104716196656,-0.306075185537,-0.406704396009,0.10038292408,0.712464630604,0.195787191391,-0.0189999304712,-0.146252155304,0.15387006104,-0.297544956207,0.317622750998,0.0184208322316,-0.128658607602,0.0909625515342,-0.0311253629625,0.260530024767,-0.622160255909,0.687025904655,-0.309245109558
```

### 生成树存储结构及格式

聚类产生的是一个编码树, 并以Key Value的形式存储在任意分布式或本地Key Value存储中.编码树的编码方式见下图.

<div align=center>
<img src="https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution/TDM/docs/tree.png" alt="编码树编码方式" />
</div>

其中Key是各个节点的编码, Value则以PB的形式存储各种属性是信息, 节点存储信息的PB描述如下.

```protobuf
message Node {
  required int64 id = 1;
  required float probality = 2;
  required int32 leaf_cate_id = 3;
  required bool is_leaf = 4;
  repeated float embed_vec = 5;
  optional bytes data = 6;
}
```

在Key Value存储之上, 树提供各种丰富的接口, 可以访问树的祖先, 兄弟, 孩子, 并能按层进行迭代. 整个设计架构如下.

<div align=center>
<img width="600" src="https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution/TDM/docs/Tree_Structure.png" alt="树存储设计架构" />
</div>


# 引用

[1] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning Tree-based Deep Model for Recommender Systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18). 1079-1088.

[2] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep Interest Network for Click-Through Rate Prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18). 1059-1068.