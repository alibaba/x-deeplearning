# XDL: 面向高维稀疏场景的工业级深度学习框架

![Travis (.org)](https://img.shields.io/badge/build-passing-green.svg)
![Travis (.org)](https://img.shields.io/badge/docs-latest-green.svg)
[![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)](https://github.com/alibaba/x-deeplearning/blob/master/LICENSE)

# 概述
* 为稀疏场景而生。支持千亿参数，万亿样本的深度模型训练，无论使用CPU训练还是GPU训练，都可以极致的压榨硬件的使用率
* 工业级分布式训练。原生支持大规模分布式训练，具备完整的分布式容灾语义，系统的水平扩展能力优秀，可以轻松做到上千并发的训练。同时内置了完整的在线学习解决方案，可以自动的进行特征选择和过期淘汰，保证在线服务的模型控制在合理的规模
* 混合多后端支持。单机内部的稠密计算复用了开源深度学习框架的能力，只需要少量的驱动代码修改，就可以把TensorFlow/MxNet的单机代码运行在XDL上，获得XDL分布式训练与高性能稀疏计算的能力
* 高效的结构化压缩训练。针对互联网样本的数据特点，提出了结构化压缩训练模式。在多个场景下，相比传统的平铺样本训练模式，样本存储空间、样本IO效率、训练绝对计算量等方面都大幅下降，训练效率可以最大可提升10倍以上
* 内置阿里妈妈广告推荐场景优秀的[算法解决方案](https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution)

# 安装
## Docker镜像(推荐方式)

XDL的Docker镜像基于[ubuntu16.04](https://hub.docker.com/_/ubuntu/)进行编译安装。镜像中中包含编译XDL所需要的所有系统环境以及编译安装完成的XDL和测试用例代码。用户也可以直接使用我们提供的Docker镜像进行XDL的二次开发。

#### XDL Docker系统要求

1. 宿主机上需要[安装docker环境](https://docs.docker.com/install/)
2. 如果需要GPU支持，宿主机上需要[安装nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

#### 下载XDL Docker镜像

XDL官方镜像的tag及描述如下表：

| docker镜像 | 描述 |
| --- | --- |
|registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12| ubuntu16.04 + XDL + tensorflow1.12后端|
|registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-gpu-tf1.12| ubuntu16.04 + XDL + tensorflow1.12后端 + gpu支持(cuda9.0, cudnn7.0)|
|registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-mxnet1.3| ubuntu16.04 + XDL + mxnet1.3后端|
|registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-gpu-mxnet1.3| ubuntu16.04 + XDL + mxnet1.3后端 + gpu支持(cuda9.0, cudnn7.0)|

例如，用户可以执行如下命令将XDL的镜像下载到宿主机上：

```
sudo docker pull registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12
```

#### 启动XDL镜像

用户可以使用以下命令，来启动一个XDL镜像：

```
sudo docker run -it registry.cn-hangzhou.aliyuncs.com/xdl/xdl[:tag] [command] 
```

更多docker的用法，可参考[docker使用指南](https://docs.docker.com/engine/reference/run/)。


例如，用户可以使用以下命令，来查看一个CPU版本的XDL镜像：

```
sudo docker run -it registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12 python -c "import xdl; print xdl.__version__"
```

用户也可以使用以下命令，来查看一个支持GPU的XDL镜像：

```
sudo nvidia-docker run -it registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-gpu-tf1.12 python -c "import xdl; print xdl.__version__"
```

请注意，上述命令执行后并不能进入docker，只是打印docker环境中安装的xdl版本。请阅读docker使用指南获取如何进入docker的方法，以及如何指定进入目录、如何挂载宿主机上需要使用的目录、如何使用网卡等。

## GPU支持

如果用户需要GPU支持，那么我们强烈建议使用支持GPU的XDL镜像。这样，用户只需要在宿主机上安装[NVIDIA® GPU drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)。

```
注意：CUDA9.0要求NVIDIA驱动版本高于384.x。
```

如果用户需要在宿主机上直接获得GPU支持，那么我们推荐用户对以下的软件版本进行安装：

* [NVIDIA® GPU drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us) (384.x or higher)
* [CUDA® Toolkit](https://developer.nvidia.com/cuda-zone) 9.0
* [cuDNN SDK](https://developer.nvidia.com/cudnn) 7.0

## 源码编译

这部分内容将指导用户如何在一个纯净的ubuntu 16.04环境中编译安装XDL。

#### 1.安装系统软件
(1) 下载ubuntu 16.04镜像，并以bash session的形式进入镜像: 
```
sudo docker pull ubuntu:16.04
sudo docker run --net=host -it ubuntu:16.04 /bin/bash
```

 系统更新:

```
apt-get update && apt-get -y upgrade
```

(2) 安装系统build工具，例如gcc、cmake、jdk、python等:
目前XDL使用seastar作为底层通信库，seastar只能使用GCC 5.3及以上版本编译。同时XDL需要使用GCC 4.8.5 编译CUDA代码来支持GPU，因此需要安装两个版本GCC; XDL使用cmake作为build工具，版本>2.8.0

```
apt-get install -y build-essential gcc-4.8 g++-4.8 gcc-5 g++-5 cmake python python-pip openjdk-8-jdk wget && pip install --upgrade pip
```

#### 2.安装依赖库
主要依赖库包括两部分，a) 安装boost b) 安装seastar依赖库

(a) 安装boost(以1.63为例), 必须设置_GLIBCXX_USE_CXX11_ABI=0，否则可能有链接错误
```
cd /tmp
wget -O boost_1_63_0.tar.gz https://sourceforge.net/projects/boost/files/boost/1.63.0/boost_1_63_0.tar.gz
tar zxf boost_1_63_0.tar.gz && cd boost_1_63_0
./bootstrap.sh --prefix=/usr/local && ./b2 -j32 variant=release define=_GLIBCXX_USE_CXX11_ABI=0 install
mkdir -p /usr/local/lib64/boost && cp -r /usr/local/lib/libboost* /usr/local/lib64/boost/
```

(b) 安装其它依赖
```
apt-get install -y libaio-dev ninja-build ragel libhwloc-dev libnuma-dev libpciaccess-dev libcrypto++-dev libxml2-dev xfslibs-dev libgnutls28-dev liblz4-dev libsctp-dev libprotobuf-dev protobuf-compiler libunwind8-dev systemtap-sdt-dev libjemalloc-dev libtool python3 libjsoncpp-dev
```
#### 3. 如果是GPU版本，还需安装cuda和cudnn，可以参考[Dockerfile](https://github.com/alibaba/x-deeplearning/blob/master/xdl/docker/Dockerfile)，主要步骤如下：

```
cd /tmp
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb
dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
apt-get clean && apt-get update -y && apt-get -y upgrade
apt-get install -y cuda=9.0.176-1 --fix-missing
apt-get install -y libcudnn7-dev
apt-get install -y libnccl-dev
apt-get update -y && apt-get -y upgrade
rm -rf /tmp/*.deb
echo '/usr/local/nvidia/lib64/' >> /etc/ld.so.conf
```

#### 4. 安装深度学习后端，目前支持TensorFlow和Mxnet
* TensorFlow
  * 从[github](https://github.com/tensorflow/tensorflow)下载tensorflow-rc1.12源码
  * 按照[官方wiki](https://www.tensorflow.org/install/source)从源码编译，并加上编译选项: --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"

* 使用Mxnet
  * 从[github](https://github.com/apache/incubator-mxnet.git)下载mxnet1.3源码
  * 按照[官方wiki](https://mxnet.incubator.apache.org/install/build_from_source.html)从源码编译，并加上编译选项：USE_CPP_PACKAGE=1
  * XDL使用mxnet C++接口执行后端计算，需要依赖其C++接口，mxnet默认不安装头文件，因此需要手动安装  

```
cp -r incubator-mxnet/cpp-package `python -c "from __future__ import print_function;import mxnet,os; print(os.path.dirname(mxnet.__file__),end='')"`
cp -r incubator-mxnet/include  `python -c "from __future__ import print_function;import mxnet,os; print(os.path.dirname(mxnet.__file__),end='')"`
cp -r incubator-mxnet/3rdparty `python -c "from __future__ import print_function;import mxnet,os; print(os.path.dirname(mxnet.__file__),end='')"`
```

#### 5. 编译XDL(以TensorFlow作为后端为例)
* 代码准备

```
git clone --recursive https://github.com/alibaba/x-deeplearning.git
cd x-deeplearning/xdl
mkdir build && cd build
export CC=/usr/bin/gcc-5 && export CXX=/usr/bin/g++-5
```

* 编译CPU版本

```
cmake .. -DTF_BACKEND=1
make -j32 
# 安装
make install_python_lib
```

* 编译GPU版本

```
cmake .. -DUSE_GPU=1 -DTF_BACKEND=1 -DCUDA_PATH=/usr/local/cuda-9.0 -DNVCC_C_COMPILER=/usr/bin/gcc-4.8
make -j32
# 安装
make install_python_lib
```

* 验证安装

在shell中执行一下命令，如果显示1.0表示安装成功

```
python -c "import xdl; print xdl.__version__"

```

# 集群部署

XDL提供了基于yarn+docker的分布式调度工具，完成集群部署后即可提交XDL分布式训练任务，具体请参考[集群部署](https://github.com/alibaba/x-deeplearning/wiki/%E9%9B%86%E7%BE%A4%E9%83%A8%E7%BD%B2)

# 快速开始

本节描述如何使用XDL进行DeepCtr(Deep+Embeddings)模型训练

### 0. 模型描述
* 示例模型包含一路Deep(deep0)特征以及两路sparse特征(sparse[0-1])，sparse特征通过Embedding计算生成两个8维的dense向量，并与Deep特征concat之后经过4层全连接层输出loss
* [样本格式](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#sample_format)
* [完整代码](http://gitlab.alibaba-inc.com/alimama-data-infrastructure/XDL-OpenSource/tree/master/xdl/examples/deepctr)

### 1. 读取数据 ([detail](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#data_io))

```
import xdl
import tensorflow as tf

reader = xdl.DataReader("r1", # reader名称                                                                                                                                                                                               
                        paths=["./data.txt"], # 文件列表                                                                                                                                                                                    
                        enable_state=False) # 是否打开reader state，用于分布式failover，开启的时候需要额外的命令行参数(task_num)                                                                                                       
reader.epochs(1).threads(1).batch_size(10).label_count(1)
reader.feature(name='sparse0', type=xdl.features.sparse)\  # 定义reader需要读取的特征，本例包括两个sparse特征组和一个dense特征组                                                                                              
    .feature(name='sparse1', type=xdl.features.sparse)\
    .feature(name='deep0', type=xdl.features.dense, nvec=256)
reader.startup()                                                                                                                                                 

```

### 2. 定义模型 ([detail](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#sparse_define))

* Embedding：

```
emb1 = xdl.embedding('emb1', # embedding名称，如果多路embedding共享同一个参数，name需配成同一个
                      batch['sparse0'], # 输入特征(xdl.SparseTensor)
                      xdl.TruncatedNormal(stddev=0.001),  # 参数初始化方法
                      8, # embedding维度
                      1024,  # sparse特征的维度
                      vtype='hash') # sparse特征类型：index(id类特征)/hash(hash特征)                                                                     
emb2 = xdl.embedding('emb2', batch['sparse1'], xdl.TruncatedNormal(stddev=0.001), 8, 1024, vtype='hash')                                                                      
```
* Dense：

```
@xdl.tf_wrapper()
def model(deep, embeddinc1 = tf.layers.dense(                                                                                                                                                        
        input, 128, kernel_initializer=tf.truncated_normal_initializer(                                                                                                             
    fc3 = tf.layers.dense(                                                                                                                                                        
        fc2, 32, kernel_initializer=tf.truncated_normal_initializer(                                                                                                              
            stddev=0.001, dtype=tf.float32))                                                                                                                                      
    y = tf.layers.dense(                                                                                                                                                          
        fc3, 1, kernel_initializer=tf.truncated_normal_initializer(                                                                                                               
            stddev=0.001, dtype=tf.float32))                                                                                                                                      
    loss = tf.losses.sigmoid_cross_entropy(labels, y)                                                                                                                             
    return loss          
```

### 3. 定义优化器 ([detail](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#optimizer))

```
loss = model(batch['deep0'], [emb1, emb2], batch['label'])                                                                                                                    
train_op = xdl.SGD(0.5).optimize()   
```

### 4. 定义训练流程 


```
log_hook = xdl.LoggerHook(loss, "loss:{0}", 10) 
sess = xdl.TrainSession(hooks=[log_hook])                                                                                                                                     
while not sess.should_stop():                                                                                                                                                 
    sess.run(train_op) 
```

### 5. 执行训练 ([detail](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#single_train))

将上述代码保存为deepctr.py，执行以下步骤开始单机训练

```
# 进入docker，将代码和数据目录一起挂载进docker
sudo docker run -v [path_to_xdl]/examples/deepctr:/home/xxx/deepctr -it registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12 /bin/bash
# 进入训练目录
cd /home/xxx/deepctr
# 开始单机训练
python deepctr.py --run_mode=local
```

# 用户指南
* 数据准备
  * [样本格式](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#sample_format)
  * [读取数据](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#data_io)
  * [python reader](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#python_reader)
* 定义模型
  * [稀疏部分](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#sparse_define)
  * [稠密部分](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#dense_define)
  * [优化器](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#optimizer)
* 训练模型
  * [单机训练](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#single_train)
  * [分布式训练](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#multi_train)
  * [同步及半同步训练](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#sync_train)
  * [保存恢复模型变量](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#checkpoint)
* 模型评估
  * [模型评估](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#evaluation)
* 高层训练API
  * [Estimator](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#estimator)
* 调试工具
 * [Timeline](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3#trace)

# Contribution
欢迎对机器学习有兴趣的同仁一起贡献代码，提交Issues或者Pull Requests，请先查阅: [XDL Contribution Guide](https://github.com/alibaba/x-deeplearning/wiki/Contributing)

# FAQ
* [常见问题](https://github.com/alibaba/x-deeplearning/wiki/FAQ)

# License
XDL使用[Apache-2.0](https://github.com/alibaba/x-deeplearning/blob/master/xdl/LICENSE)许可

