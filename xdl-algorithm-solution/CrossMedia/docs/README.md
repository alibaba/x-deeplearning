# CrossMedia Networks(Deep Image CTR Model)

# 算法介绍
CrossMedia Networks(论文里称作Deep Image CTR Model， DICM) 由阿里妈妈精准定向算法团队在CIKM2018论文中提出：
[Image Matters: Visually modeling user behaviors using Advanced Model Server](https://arxiv.org/abs/1711.06505)它旨在应用**图像信息**对广告和用户行为进行理解和表达，帮助电商广告点击率（CTR）预估。

精准展示广告系统为每次投放请求，根据用户特质、投放环境选择收益最大化的广告投放，选择流程分为匹配（Match）与初排序（Pre-rank）和精细排序（rank）三个由粗到细的步骤, 代码中我们给出CrossMedia Network在精细排序中点击率预估的应用，在初排序中也可以使用类似方法。

在电商场景中，广告是以图片的形式展示给用户，图像信息对于广告点击率有直接影响。另一方面，用户行为的准确理解直接关系到广告的个性化和智能化程度。 因此，我们不仅用图像表达广告，也在用户行为理解中添加了图像信息。在CrossMedia 工作中：

1. 网页中展示、作为点击入口的图像作为广告图像（Ad image）表达广告

2. 用户过去14天点击过的商品图像作为用户行为图像（User behavior images）表达用户行为

广告图像和用户行为图像与原有的ID类特征共同参与到广告CTR预估网络，联合训练优化，是CrossMedia的特色。

![网络结构与训练系统](http://git.cn-hangzhou.oss-cdn.aliyun-inc.com/uploads/tiezheng.gtz/XDL-Algorithm-Solution/6d8804cf5ea77b505f096c7ad8132732/ams.png)

## 网络结构 Deep Image CTR Model（DICM） 

CrossMedia 的网络结构如上图左边DICM所示。和目前主流方法一致，我们将点击率预估问题看作对于每个广告展示样本的点击或未点击的二元判别问题。我们以流行的Emedding&MLP 为基础网络，即所有ID特征（可能为on-hot或multi-hot）Embed为向量，再经过向量拼接、多层全连接网络得到点击/未点击的二维向量输出。

DICM网络在基础网络之上，加上了广告图像和用户行为图像部分，广告图像经过图像Embed网络得到“广告图像向量”，同样, 用户行为图像也各自经过Embed网络得到向量，再经过聚合器聚合成“用户行为图像向量”。“广告图像向量”、“用户行为图像向量” 和ID特征得到的向量一起拼接，经过全连接网络得到输出

在DICM中，图像Embed网络被设计为“固定网络”与“可训练网络”两部分（如下图所示），“固定网络”成熟VGG16模型不参与训练，“可训练“网络为全连接网络参与DICM联合训练。兼顾了性能和效率。而用户行为图像的聚合器可采用简单的按维度相加Pooling，也可以引入注意力Pooling取得更好的效果。

<figure>
    <img src="http://git.cn-hangzhou.oss-cdn.aliyun-inc.com/uploads/tiezheng.gtz/XDL-Algorithm-Solution/e671d149e72d6925b80fb63aece01671/embed.png"title="embed" width="500">
</figure>

## 训练系统 Advanced Model Server（AMS）

CrossMedia 的主要创新体现在其训练系统。由于图像容量高，在以十亿计大规模数据引入图像特征（尤其是用户行为图像）将带来计算、存储、通信的沉重负担。因此，XDL专门开发了Advanced Model Server（AMS）处理图像Embed网络。

如上面大图右边AMS部分所示，AMS在以参数服务器（PS）为代表的分布式计算架构基础上再前进一步，将模型训练的一部分网络由worker部分移到了Server部分，也就是说，改造后的Server 也具有模型前向、后向计算的功能。在CrossMedia中：

1. 图像和其他ID参数一样，分布式存储在Server里边，Worker需要时想Server请求，避免重复存储
2. 图像Embed网络计算也放在Server里边（AMS的特有功能），图像被请求后，就地计算Embed，得到embed 向量传到Worker，这样的改造减少了Worker与Server的通讯，也去掉了同一Batch中请求同一图像的重复Embed计算

值得注意的是，AMS也可以处理各种类似的需要Embed的内容特征，如文本、视频。




# CrossMedia训练流程

## 数据准备
由于图片数据权限问题，我们无法提供数据集，需要自行自行准备数据

输入数据格式：
## XDL系统
在XDL系统中，我们实现了CrossMedia的训练模式。
如果希望使用下述的解决方案，需要先部署XDL系统。
具体请见[XDL系统的介绍]()。

## 简单例子
我们提供了一个简单的CrossMedia实现供试验用。代码请见script/run.sh

简单地在一个部署了XDL的集群中运行```sh run.sh```即可运行一个简单的CrossMedia任务。

我们提供了randtxt.py用于样本输入的构建，randimg.py用于图片输入的构建。而config.json和runner.py用于训练。
后面的章节将分别讲述这些部分如何使用。

## 数据准备
在XDL中提供了基础的数据格式。你可以选择简单地遵循这些数据格式。

对于样本数据，XDL提供了data_io的接口，可以满足多种数据格式的要求。
具体情况可见[data_io文档]()。
randtxt.py依照这种格式生成了样本数据，你也可以仿照这个代码生成样本数据，需要注意，如果数据过多你可以把数据放置在hdfs上。

同样对于图像数据，我们提供了[DataSource接口]()。
DataSource实际上只实现了一个Key-Value存储结构，你可以把图像数据以图片id为Key，图片数据为Value存储于其中。
同样的，你也可以仿照randimg.py去生成图像数据，并放置在hdfs上。

## 训练
CrossMedia的训练流程与普通的XDL任务基本一致，
你可以先查看[XDL用户文档]()来获取一些基础信息。

CrossMedia独特之处在于其有一个特殊的角色：ModelServer。ModelServer将负责将图片数据做分割，分不同id区域进行计算。
你可以查阅[ModelServer的文档]()了解更多信息。

## Cite Bib
```
@inproceedings{ge2018image,
  title={Image Matters: Visually Modeling User Behaviors Using Advanced Model Server},
  author={Ge, Tiezheng and Zhao, Liqin and Zhou, Guorui and Chen, Keyu and Liu, Shuying and Yi, Huimin and Hu, Zelin and Liu, Bochao and Sun, Peng and Liu, Haoyu and others},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={2087--2095},
  year={2018},
  organization={ACM}
}
```