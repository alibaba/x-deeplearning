# 环境准备
### 1. 安装XDL环境(推荐使用docker方式运行XDL提供的ubuntu镜像: registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12)
### 2. 数据准备
* 按照data/README.md的步骤得到DIN使用的数据文件，包括以下7个文件:
  * cat_voc.pkl
  * mid_voc.pkl
  * uid_voc.pkl
  * local_train_splitByUser
  * local_test_splitByUser
  * reviews-info
  * item-info
3. 配置config.json中的data_dir选项为数据实际存储路径，比如../data/

# 单机训练
### 1. 在宿主机上安装docker

### 2. 进入docker镜像，并将对应算法目录挂载进docker内:

```
sudo docker run --net=host -v [path_to_din]:/home/xxx/DIN -it registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12 /bin/bash
```

### 3. 在docker中执行以下命令开始单机训练:

```
cd /home/xxx/DIN/script
python train.py --run_mode=local --config=config.json
```

# 效果验证
* train.py会使用训练和评估数据(local_train_splitByUser和local_test_splitByUser)交替执行训练和评估，auc会输出到stdout