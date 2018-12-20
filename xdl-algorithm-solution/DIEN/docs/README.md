# Deep Interest Evolution Network for Click-Through Rate Prediction

DIEN (Deep Interest Evolution Network for Click-Through Rate Prediction) 主要用于定向广告排序中的CTR（点击率）预估阶段。在CTR预估模型进入深度时代后，相关的工作层出不穷，如Wide&Deep，PNN，DeepFM。
但是这些模型都局限于Embedding&MLP的范式，缺乏了对CTR预估真实应用情景的思考, DIN针对电商场景提出了更好的兴趣建模方案。但是这些工作都直接将用户的具体历史行为当做了用户某一时刻的兴趣。
这篇工作中我们提出用户的兴趣是一个抽象的概念，用户的历史行为只是抽象的兴趣的一个具体的体现。
在DIEN中我们提出了兴趣抽取和兴趣演化两个模块共同组成的CTR预估模型。该算法被应用于阿里妈妈定向广告各大产品中，在DIN的基础上取得了非常显著的效果提高。

DIEN 主要面向两个问题：兴趣提取和兴趣演化。
在兴趣提取这部分，传统的算法直接将用户的历史行为当做用户的兴趣。同时整个建模过程中的监督信息全部集中于广告点击样本上。而单纯的广告点击样本只能体现用户在决策是否点击广告时的兴趣，很难建模好用户历史每个行为时刻的兴趣。本文中我们提出了auxiliary loss 用于兴趣提取模块，约束模型在对用户每一个历史行为时刻的隐层表达能够推测出后续的行为，我们希望这样的隐层表达能更好的体现用户在每一个行为时刻的兴趣。
在兴趣提取模块后我们提出了兴趣演化模块，传统的RNN类似的方法只能建模一个单一的序列，然而在电商场景 用户不同的兴趣其实有不同的演化过程。在本文中我们提出了AUGRU（Activation Unit GRU），让GRU的update门和预估的商品相关。在建模用户的兴趣演化过程中，AUGRU会根据不同的预估目标商品构建不同的兴趣演化路径，推断出用户和此商品相关的兴趣。
![undefined](https://cdn.nlark.com/lark/0/2018/png/36154/1541996406807-37ca2fb3-77e1-4c05-9d71-7916ed96da62.png) 

#### 兴趣提取层
兴趣提取层部分我们主要采用GRU结构来对用户行为序列进行建模，获取得到用户在不同时刻的兴趣表达。
同时我们在每个时间点约束当前兴趣表达可以预测下一个时刻的点击以及用户下时刻采样的不点击行为。我们将这样的约束方式作为模型的辅助loss的方式引入学习。通过加入辅助loss的方式不仅能够引入用户的反馈信息并且还能够帮助长序列的学习，降低梯度回传难度，同时还能够提供更多的语义信息帮助embedding部分的学习。

#### 兴趣演化层
用户的兴趣是多种多样的，其同时存在多个兴趣轨迹，我们在预测当前AD时，只需要关心和这个目标AD相关的兴趣的演化状态。在DIN算法里我们采用的是attention的方式得到用户和当前ad相关的兴趣状态，但是没有考虑到用户兴趣间的演化关系。所以我们在兴趣演化层部分首先将和当前ad相关的子兴趣提取出来，然后把这些子兴趣进行序列建模，从而能够获取得到和当前ad相关的兴趣演化信息。
在这里我们将GRU结构进行了改进，将ad和兴趣的相关信息引入了门更新，实现了对不同的目标AD，用户都有一条独有的兴趣演化轨迹。

![CodeCogsEqn-5.svg](https://cdn.nlark.com/lark/0/2018/svg/6098/1542348078892-ef6f1ebd-bf59-4378-af7e-30a93c712e0d.svg) 

其中 ![img](http://latex.codecogs.com/svg.latex?a_t)是ad和当前时间点兴趣(由兴趣提取层提取得到)的相关度权重

![CodeCogsEqn-6.svg](https://cdn.nlark.com/lark/0/2018/svg/6098/1542348337446-cd5b5a5e-f378-4af7-83b5-441b4bf64fcc.svg) 

我们将兴趣演化层最后一个时刻的兴趣表达 ![img](http://latex.codecogs.com/svg.latex?h'(T))输出作为用户兴趣。![img](http://latex.codecogs.com/svg.latex?h'(T))$捕捉到了用户兴趣的演化信息并且是和ad相关的子兴趣表达。最后将![img](http://latex.codecogs.com/svg.latex?h'(T))和ad特征、上下文特征、用户静态特征拼接一起，输入到多层dnn网络中进行预测。

# DIEN训练流程
* 详见script/README.md

## Cite Bib
```
@inproceedings{zhou2018deep,
  title={Deep interest evolution network for click-through rate prediction},
  author={Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, Kun Gai},
  booktitle={Proceedings of the 33nd AAAI Conference on Artificial Intelligence},
  year={2019},
  organization={AAAI}
}
```