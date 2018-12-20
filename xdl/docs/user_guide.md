# 使用指南
* 特征体系
  * [样本](#sample_define)
  * [特征](#feature_define)
  * [网络](#network_example)
* 数据准备
  * [样本格式](#sample_format)
  * [使用data_io读取数据](#data_io)
  * [自定义python reader](#python_reader)
* 定义模型
  * [稀疏部分](#sparse_define)
  * [稠密部分](#dense_define)
  * [优化器](#optimizer)
* 训练模型
  * [单机训练](#single_train)
  * [分布式训练](#multi_train)
  * [同步及半同步训练](#sync_train)
  * [保存及恢复模型变量](#checkpoint)
* 模型评估
  * [模型评估](#evaluation)
* 高层训练API
  * [Estimator](#estimator)
* 调试
  * [Timeline](#trace)
  
# 0. 特征体系

<a name="sample_define"></a> 
### 0.1 样本

一条样本由 label, feature, sampleid 三部分组成；训练过程是通过feature学习label的过程，sample id是一条样本的标识。label可以是单值(二分类)或者多值(多分类)。  

<a name="feature_define"></a> 
### 0.2 特征

传统的特征一般是稠密的(dense)，维度不会特别高，每条样本都出现；以向量的形式表示；例如图像特征，或者用户的性别信息。  
对于搜索-推荐-广告场景，存在大量的稀疏特征(sparse)：这些特征维度高（百亿），但是每条样本中出现次数低（数百)，这类特征以多个kv的方式稀疏表示；例如商品类目(key是类目的id，没有value)，用户点击过的商品列表(key是商品id，value是点击次数)。  

特征以特征组的方式组织，便于构建训练网络；例如特征分为：用户性别(稠密特征)，用户偏好(稀疏特征)，商品类目(稀疏特征)，商品价格(稠密特征)；

<a name="network_example"></a> 
### 0.3 网络

简单的稀疏场景训练网络举例如下：

![io_compact_network](http://git.cn-hangzhou.oss-cdn.aliyun-inc.com/uploads/alimama-data-infrastructure/XDL/c28e86a13945a3124bb37e757ce5f9a0/io_compact_network.png)


# 1. 数据准备
<a name="sample_format"></a> 
### 1.1 样本格式

#### 1.1.2 文本格式

文本格式一行表示一条样本，分为多个字段，用'|'分隔
字段定义和分隔符如下：

|field|desc|value|example|
|---|---|---|---|
|sample id|sample的唯一描述，用于调试|string|7859345_420968_1007|
|group id|样本组的标识，连续一样的会聚合到一个样本组|string|user_3423487|
|sparse|稀疏特征，用kv表示|多组特征用';'分隔，一个特征名字和内存用'@'分隔，内容多值用','分隔，稀疏的key和value用':'分隔|clk_14@32490:1.0,32988:2.0;prefer@323423,32342|
|dense|稠密特征|多组特征用';'分隔，一个特征内多值用','分隔|qscore@0.8,0.5;ad_price@33.8|
|label|目标|float, 多值用','分隔|0.0,1.0|
|ts|时间戳|int|1544094136|

```
# sample id        | group id   | sparse                                       | dense                      | label | timestamp
7859345_420968_1007|user_3423487|clk_14@32490:1.0,32988:2.0;prefer@323423,32342|qscore@0.8,0.5;ad_price@33.8|0.0,1.0|1544094136

```

#### 1.1.3 protobuf格式

* 见[结构化压缩](#structure_computing)


<a name="data_io"></a> 
### 1.2 使用data_io读取数据

#### 1.2.1 初始化和销毁

定义模型前，需要初始化data io，设置选项，start up线程

python脚本结束后, data io会等待所有线程退出后销毁

```
io = DataIO(name, file_type, fs_type, name_node="", enable_state=False)

# setup options
io.epochs(1)
io.batch_size(1024)
io.label_count(2)
io.feature(...)
# ...

io.startup()
```
|option|desc|example|
|---|---|---|
|name|一个数据源的唯一标识|"dnn"|
|file_type|数据文件格式|xdl.parsers.pb/txt|
|fs_type|文件系统类型|xdl.fs.local/hdfs/kafka|
|name_node|文件系统namenode|"hdfs://xxx/xxx"|
|enable_state|保存状态，failover将从上个保存点回复|True/False|

#### 1.2.2 选项设置

|option|desc|default|example|
|---|---|---|---|
|epochs|设置样本遍历的epoch数目，0表示一直循环|1|io.epochs(4)|
|batch_size|设置batch size|1024|io.batch_size(1024)|
|label_count|设置label的维数|2|io.label_count(2)|
|threads|设置线程数，文件平均分配给各个线程|1|io.threads(8)|
|add_path|添加文件，可以是hdfs地址或kafka的topic||io.add_path('hdfs://xxx/xxx')|
|add_op|添加io-op，可以从so加载||de=xdl.GetIOP("DebugOP"); io.add_op(de)|
|keep_sample|保留使用过的样本作为下一次重新采样的种子，配合op使用|False|io.keep_sample(True)|
|split_group|batching的时候拆分sample group|True|io.split_group(False)|
|unique_ids|在io阶段流水线异步计算稀疏特征的uniq id|False|io.unique_ids(True)|
|pause|读N个sample group后暂停，配合keep_sample使用||io.pause(1024, True)|
|feature|设置需要使用的feature||
| |type: 稀疏或者稠密特征||io.feature(name="xxx", type=FeatureType.sparse)|
| |table: 结构化压缩的表，不压缩是0|0|io.feature(name="xxx", type=FeatureType.sparse, table=0)|
| |serialized: 序列化特征，64bit表示，非序列化128bit表示|False|io.feature(name="xxx", type=FeatureType.sparse, serialized=True)|
| |dsl:自动交叉特征描述||io.feature(name="xxx", type=FeatureType.sparse, dsl="match(feature1, feature2)")|

### 1.2.3 数据并行DataSharding

dataio可以使用add_path加入数据文件、hdfs路径或kafka topic

当文件比较多，或者需要多机sharding的时候，更方便的办法是使用DataSharding，可以添加目录，支持正则表达式，并按机器数划分文件

```
io = xdl.DataIO("tdm", file_type=xdl.parsers.txt, fs_type=xdl.fs.hdfs, namenode='hdfs://xxx, enable_state=True)

sharding = xdl.DataSharding(io.fs())
sharding.add_path(r"hdfs://xxx/sample[.\w]+$")

paths = sharding.list()

paths = sharding.partition(rank=xdl.get_task_index(), size=xdl.get_task_num())

io.add_path(paths)
```
<a name="structure_computing"></a> 
### 1.2.3 结构化压缩

结构化压缩是指，多个样本中共同的特征值，只存储一份，也只进行一次计算；需要在样本处理中，聚合多条具有共同特征的样本

例如广告样本中，一个用户可能点击N个广告，产生N条样本；这N条样本的广告特征簇都是不同的，但是用户特征簇都是相同的一份

其存储和计算过程示意如下：

![io_compact_pipeline](http://git.cn-hangzhou.oss-cdn.aliyun-inc.com/uploads/alimama-data-infrastructure/XDL/671462225d5b732ce9ea90341d57c9e8/io_compact_pipeline.png)

使用结构化压缩，需要在pb样本中表达特征结构化，主要是通过定义多个特征表

例如下图定义了一个特征主表(ad)和一个特征辅表(user)，通过主表的多个ad特征指向辅表的一个user特征，表示多个样本的特征复用关系

下图中，a0 和 a1 特征共用一个u0特征；a2, a3, a4共用一个u2特征

![io_compact_pb](http://git.cn-hangzhou.oss-cdn.aliyun-inc.com/uploads/alimama-data-infrastructure/XDL/667b85622d93141a704355c23277a8e3/io_compact_pb.png)

特征表的定义是一个repeated， 这样可以通过定义两个以上的特征表，来表示多层压缩。例如 图片 -> 广告 -> 用户 这样两层的多对一关系

```
message SampleGroup {
    repeated string sample_ids = 1;            // 每个样本的sample_id
    repeated Label labels = 2;                 // 每个样本的label， Label类型
    repeated FeatureTable feature_tables = 3;  // 整个sample的特征表，如果没有辅表，只有一个feature_table
    repeated Label props = 4;                  // 每个样本的predict结果， Label类型
    repeated Extensions extensions = 5;        // 每个样本的扩展字段，待以后扩展

}

message Extensions {
    map<string, string> extension = 1;
}

// 标签，支持多目标训练
message Label {
    repeated float values = 1;
}

// 特征表
message FeatureTable {
    repeated FeatureLine feature_lines = 1; // 每个样本的特征行
}

// 特征行
message FeatureLine {
    repeated Feature features = 1;         // 每个特征行里的特征(组)
    optional int32 refer = 2;              // 引用下层辅表的哪个特征行
}

// 特征(组)
message Feature {
    required FeatureType type = 1;         // 特征类型
    optional string name = 2;              // 特征(组)名字，与field_id二选一
    repeated FeatureValue values = 3;      // 特征值, 一个特征(组)可能有多个特征值
}

// 特征值
message FeatureValue {
   optional int64 key = 1;                 // 特征ID, dense可以没有
   optional float value = 2;               // 特征值，没有默认是1
   repeated float vector = 3;              // 特征向量，向量类型的特征才有，也可以用来表示稠密特征
   optional int64 hkey = 4;                // 特征ID高64位，用来支持128位hashkey
}
```

<a name="python_reader"></a> 
### 1.4 自定义python reader

* xdl支持直接使用python定义op

```
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('./data')

# python读取函数，直接使用tf封装好的api读取mnist数据
def read_data(batch_size=100):                                                                                                                                                                                                               
    global mnist_data                                                                                                                                                                                                                         
    images, labels = mnist_data.train.next_batch(batch_size)                                                                                                                                                                                  
    labels = np.asarray(labels, np.float32)                                                                                                                                                                                                   
    return images, labels      

# 通过xdl.py_func定义op
images, labels = xdl.py_func(read_data, [], output_type=[np.float32, np.float32])        

```

# 2. 定义模型

XDL专注解决搜索广告等稀疏场景的模型训练性能问题，因此将模型计算分为稀疏和稠密两部分，稀疏部分通过参数服务器，GPU加速，参数合并等技术极大提升了稀疏特征的计算和通信性能。稠密部分采用多backend设计，支持TF和Mxnet两个引擎作为计算后端，并且可以使用原生TF和Mxnet API定义模型。下面分别介绍稀疏和稠密部分的API

<a name="sparse_define"></a> 
### 2.1 稀疏API

* API列表

| API | 描述 |
| --- | --- |
| xdl.embedding | 计算单路稀疏特征的embedding |
| xdl.merged_embedding | 同时计算多路稀疏特征的embedding，内部将通信和计算做了合并，建议embedding较多时使用 |

* 参数说明

xdl.embedding

| 参数 | 说明 |
| --- | --- |
| sparse_input | embedding输入，是一个SparseTensor|
| initializer | embedding参数初始化方法 |
| emb_dim | embedding之后的维度 |
| feature_dim | embedding前的维度 |
| combiner | reduce方法，支持sum/mean |
| vtype | 参数类型：index/hash |

xdl.merged_embedding

| 参数 | 说明 |
| --- | --- |
| sparse_input | embedding输入，一个SparseTensor列表|
| initializer | embedding参数初始化方法 |
| emb_dim | embedding之后的维度 |
| feature_dim | embedding前的维度 |
| combiner | reduce方法：sum/mean |
| vtype | 参数类型：index/hash |

<a name="dense_define"></a> 

### 2.2 稠密API
XDL使用TF和Mxnet作为计算后端，并且支持使用TF和Mxnet原生API来定义模型
* 定义方法
  * 1. 使用TensorFlow或者Mxnet定义模型
  * 2. 使用xdl.tf_wrapper或者xdl.mxnet_wrapper修饰模型定义函数

* 装饰器参数

| 参数 | 说明 |
| --- | --- |
| is_traning | 标识当前任务是训练还是预测，训练会添加backprop |
| device_type | 设备类型：CPU/GPU |

* 使用TF Backend定义一个embedding + 5层dense网络

```
@xdl.tf_wrapper()
def model_fn(dense, embeddings, labels):
  input_features = [dense]
  input_features.extend(embeddings)
  inputs = tf.concat(input_features, 1)
  fc1 = tf.layers.dense(inputs, 256, activation=tf.nn.relu)
  fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu)
  fc3 = tf.layers.dense(fc2, 64, activation=tf.nn.relu)
  fc4 = tf.layers.dense(fc3, 32, activation=tf.nn.relu)
  logits = tf.layers.dense(fc4, 1, activation=tf.nn.relu)
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  loss = tf.reduce_mean(cross_entropy)
  return loss
```
 
* 使用Mxnet Backend定义一个embedding + 5层dense网络

```
@xdl.mxnet_wrapper()
def model_fn(dense, embeddings, label):
  input_features = [dense]
  input_features.extend(embeddings)
  inputs = mx.symbol.concat(*input_features, dim=1)
  fc1 = mx.sym.FullyConnected(data=inputs, num_hidden=256, name='fc1')
  fc2 = mx.sym.FullyConnected(data=fc1, num_hidden=128, name='fc2')
  fc3 = mx.sym.FullyConnected(data=fc2, num_hidden=64, name='fc3')
  fc4 = mx.sym.FullyConnected(data=fc3, num_hidden=32, name='fc4')
  fc5 = mx.sym.FullyConnected(data=fc4, num_hidden=1, name='fc5')
  prop = mx.symbol.SoftmaxOutput(data=fc5, label=label)
  loss = - mx.symbol.sum(mx.symbol.log(prop) * label) / 4
  return loss
```

<a name="optimizer"></a> 
### 2.3 优化器

* XDL支持常用的optimizer，包括
  * SGD  
  * Momentum 
  * Agagrad 
  * Adam 
  * Ftrl 


* 使用方法

```
optimizer = xdl.SGD(0.5)
train_op = optimizer.optimize()
sess = xdl.TrainSession()
sess.run(train_op)
```

# 3. 训练模型
XDL支持单机及分布式两种训练模式，单机模式一般用来做早期模型的调试和正确性验证，为了充分发挥XDL的稀疏计算能力，建议使用分布式模式进行大规模并行训练

<a name="single_train"></a> 

### 3.1 单机训练

XDL通过Local PS的方式支持单机训练，只需运行时给python脚本加上--run_mode=local的命令行参数即可：

```
python test.py --run_mode=local
```

如果用户需要使用XDL镜像进行单机训练，则需要先以bash session形式进入镜像，再启动命令：

```
sudo docker run -it registry.cn-hangzhou.aliyuncs.com/xdl/xdl[:tag] /bin/bash
python test.py --run_mode=local
```

<a name="multi_train"></a> 
### 3.2 分布式训练
* XDL通过ams存储参数，从而支持了分布式训练，在进行分布式训练时，需要启动ams进程和worker进程，ams进程包括一个scheduler和多个server，用户可以通过手动方式启动，也可以使用XDL提供的基于yarn+docker的分布式调度工具xdl_sumbit启动

#### 3.2.1 通过手工方式启动分布式任务

以下命令都默认宿主机上具有XDL运行环境，或用户以bash session形式进入XDL镜像。

* 启动ams-scheudler

```
# 参数解释：
  ps_cpu_cores和ps_memory_m：分给ams-server的cpu和内存资源，将会影响ams的参数分配算法
  ckpt_dir：checkpoint地址，目前支持本地和HDFS两种文件系统

python test.py --task_name=scheduler --zk_addr=zfs://xxx --ps_num=2 --ps_cpu_cores=10 --ps_memory_m=4000 --ckpt_dir=hdfs://xxx/checkpoint
```

* 启动ams-server

```
python test.py --task_name=ps --task_index=0 --zk_addr=zfs://xxx
python test.py --task_name=ps --task_index=1 --zk_addr=zfs://xxx
```

* 启动worker

```
python test.py --task_name=worker --task_index=0 --task_num=4 --zk_addr=zfs://xxx 
python test.py --task_name=worker --task_index=1 --task_num=4 --zk_addr=zfs://xxx 
python test.py --task_name=worker --task_index=2 --task_num=4 --zk_addr=zfs://xxx 
python test.py --task_name=worker --task_index=3 --task_num=4 --zk_addr=zfs://xxx 
```

#### 3.2.2 通过xdl_submit启动分布式任务
* 使用xdl_submit需要在机器上提前部署相关环境，部署方法参见[集群部署](https://github.com/alibaba/x-deeplearning/blob/master/docs/cluster_deploy.md)

* xdl_submit任务基础配置示例

```
{ 
  "job_name": "xdl_test",
  "dependent_dirs": "/home/xdl_user/xdl_test/",
  "script": "test.py",
  "docker_image": "registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12",
  "worker": {
    "instance_num": 10,
    "cpu_cores": 4,
    "gpu_cores": 0,
    "memory_m": 1000
  },
  "ps": {
    "instance_num": 2,
    "cpu_cores": 2,
    "gpu_cores": 0,
    "memory_m": 1000
  },
  "checkpoint": {
    "output_dir": "hdfs://ns1/data/xdl_user/xdl_test/checkpoint"
  }
}
```

* 基础配置项说明

| name        | default           | comment  |
| ------------- |-------|-----------|
| job_name      |  | 任务名称 |
| docker_image |      |  任务所用的docker镜像名称 |
| dependent_dirs |       |    脚本及其资源本地目录(提交任务所在机器)，会将整个目录上传到所有节点(worker/ps)上，并且节点上会将此目录作为当前工作目录 |
| script |       |    入口脚本文件，可用相对路径 |
| worker |       |    worker进程配置 |
| ps |       |    ps进程配置 |
| worker/ps.instance_num |   4    |    进程数 |
| worker/ps.cpu_cores |   6    |    cpu个数，不能超过单机最大数 |
| worker/ps.gpu_cores |   0    |    gpu个数，不能超过单机最大数 |
| worker/ps.memory_m |   4096    |    内存，单位M |
| checkpoint |      |    模型保存与恢复(checkpoint)相关配置 |
| checkpoint.output_dir |      |    checkpoint输出目录 |

* xdl_submit任务高级配置示例

```
{ 
  "job_name": "xdl_test",
  "dependent_dirs": "/home/xdl_user/xdl_test/",
  "script": "test.py",
  "docker_image": "registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12",
  "worker": {
    "instance_num": 10,
    "cpu_cores": 4,
    "gpu_cores": 0,
    "memory_m": 1000
  },
  "ps": {
    "instance_num": 2,
    "cpu_cores": 2,
    "gpu_cores": 0,
    "memory_m": 1000
  },
  "checkpoint": {
    "output_dir": "hdfs://ns1/data/xdl_user/xdl_test/checkpoint"
  },
  "bind_cores": "true",
  "scheduler_queue": "default",
  "min_finish_worker_rate": 90,
  "max_failover_times": 20,
  "max_local_failover_times": 3,
  "auto_rebalance": {
     "enable": "true"
  },
  "extend_role":{
    "ams": {
      "instance_num": 10,
      "cpu_cores": 8,
      "gpu_cores": 0,
      "memory_m": 8000,
      "script": "ams.py"
    }
  }
}
```

* 2. 高级配置项说明

| name        | default           | comment  |
| ------------- |-------|-----------|
| bind_cores | false      |  表示启动docker时是否以绑核形式启动 |
| scheduler_queue | default      |    yarn调度queue |
| min_finish_worker_rate |  90      |   整个Job Finish需要的最少worker完成比例，默认90%。 |
| max_failover_times |  20      |   整个Job总failover最大次数 |
| max_local_failover_times |  3      |   每个节点(ps/worker/extend_role)本地failover最大次数 |
| auto_rebalance |  | 参数动态优化分配开关，在大规模高并发场景下建议将enable选项设置为true，可大幅提高性能 |
| extend_role | | 扩展角色支持，可支持除了ps, worker, scheduler之外的任意多种角色调度，每种角色支持以不同的脚本启动 |

* 提交任务
   + 将test.py放到/home/xdl_user/xdl_test/目录下，如果有其他脚本或者本地数据和配置也可以放到该目录下，xdl_submit会自动将其挂载到docker内 
   + 执行命令: xdl_submit.py --config=xdl_test.json

<a name="sync_train"></a> 
### 3.3 同步及半同步训练

* 同步训练

```
#创建session时，添加同步训练的hook
hooks = []
hooks.append(xdl.SyncRunHook(xdl.get_task_index(), xdl.get_task_num()))
sess = xdl.TrainSession(hooks)
while not sess.should_stop(): 
  sess.run(train_ops)  
#sess run结束后，需要调用worker_report_finish_op
xdl.worker_report_finish_op(np.array(xdl.get_task_index(),dtype=np.int32))

```

* 半同步训练

```
#创建session时，添加半同步训练的hook，staleness为不同worker间允许的最大差异step数，默认值为0
hooks = []
hooks.append(xdl.SemiSyncRunHook(xdl.get_task_index(), xdl.get_task_num(), staleness=0))
sess = xdl.TrainSession(hooks)
while not sess.should_stop(): 
  sess.run(train_ops)  
#sess run结束后，需要调用worker_report_finish_op
xdl.worker_report_finish_op(np.array(xdl.get_task_index(),dtype=np.int32))
```

<a name="checkpoint"></a> 
### 3.4 保存和恢复模型变量
* 保存模型变量
  + 通过Saver保存

     ```
import xdl
saver = xdl.Saver()
checkpoint_version = "xxx" # checkpoint名称，一般是global step
saver.save(version = checkpoint_version)  
     ```

  + 通过CheckpointHook保存

     ```
import xdl
train_op = ...
hook = xdl.CheckpointHook(save_interval_step=1000) # 每1000个global step保存一次
sess = xdl.TrainSession(hooks=[hook])
sess.run(train_op)
     ```

* 恢复模型参数
  + 通过Saver恢复

     ```
import xdl
saver = xdl.Saver()
checkpoint_version = "xxx"
saver.restore(version = checkpoint_version)
     ```

<a name="evaluation"></a> 

# 4. 模型评估
* 模型评估是用指标反映模型在实际数据中的表现，是在训练中调整超参数，评估模型效果的重要依据。XDL提供了计算auc的默认op实现，用户也可以通过python或者c++定制自己的metrics实现

```
import xdl
saver = xdl.Saver()
saver.restore(ckpt_version)
labels = ...
predictions = ...
auc = xdl.auc(predictions, labels)
sess = xdl.TrainSession()
print sess.run(auc)
```

<a name="estimator"></a> 

# 5. 高层训练API：Estimator
* 为了简化用户编写模型训练脚本的工作量，XDL提供了Estimator API，可以允许用户以一套代码执行训练/预测/评估/训练&评估等多种类型的任务

### 使用步骤
* 1. 定义输入function

```
# 定义train输入
def input_fn():
    ...
    return feature_list, labels

# 定义predict/evaluate输入
def eval_input_fn():
    ...
    return test_feature_list, test_labels
```

* 2. 定义模型

```
@xdl.tf_wrapper()
def model_fn(feature_list, labels):
    logits = ...
    loss = ...
    return loss, logits
```
* 3. 创建Estimator

```
estimator = xdl.Estimator(model_fn=model_fn, optimizer=xdl.SGD(0.5))
```

* 4. 进行train|evaluate|predict|train&&evaluate

```
# 训练
estimator.train(input_fn, max_step=2000, checkpoint_interval=1000)

# 评估: checkpoint_version=""表示从最后一个checkpoint读取参数
estimator.evaluate(eval_input_fn, checkpoint_version="", max_step=2000)

# 预测
estimator.predict(eval_input_fn, checkpoint_version="", max_step=2000)

# 训练和评估交替执行
estimator.train_and_evaluate(train_input_fn=input_fn,
                             eval_input_fn=eval_input_fn,
                             eval_interval=1000,
                             eval_steps=200,
                             checkpoint_interval=1000,
                             max_step=5000)
```
   

<a name="trace"></a>
# 6. Timeline

### 使用步骤

* 1. 在训练中产出timeline

```
run_option = xdl.RunOption()                                                                                                                                                                                                              
run_option.perf = True                                                                                                                                                                                                                    
run_statistic = xdl.RunStatistic()                                                                                                                                                                                                        
_ = sess.run(train_ops, run_option, run_statistic)                                                                                                                                                                                   
xdl.Timeline(run_statistic.perf_result).save('./timeline.json')      

```

* 2. 在chrome中输入[chrome://tracing](chrome://tracing)，加载timeline.json

