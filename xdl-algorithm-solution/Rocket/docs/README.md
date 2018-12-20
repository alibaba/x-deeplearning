# Rocket Training

工业上，一些在线模型，对响应时间提出非常严苛的要求，从而一定程度上限制了模型的复杂程度。模型复杂程度的受限可能会导致模型学习能力的降低从而带来效果的下降。目前有2种思路来解决这个问题：一方面，可以在固定模型结构和参数的情况下，用计算数值压缩来降低inference时间，同时也有设计更精简的模型以及更改模型计算方式的工作，如Mobile Net和ShuffleNet等工作；另一方面，利用复杂的模型来辅助一个精简模型的训练，测试阶段，利用学习好的小模型来进行推断。这两种方案并不冲突，在大多数情况下第二种方案可以通过第一种方案进一步降低inference时间，同时，考虑到相对于严苛的在线响应时间，我们有更自由的训练时间，有能力训练一个复杂的模型，所以我们采用第二种思路，来设计了我们的方法。

## Our Approach
![undefined](https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution/Rocket/docs/structure.jpg)

如图所示，训练阶段，我们同时学习两个网络：Light Net 和Booster Net, 两个网络共享部分信息。我们把大部分的模型理解为表示层学习和判别层学习，表示层学习的是对输入信息做一些高阶处理，而判别层则是和当前子task目标相关的学习，我们认为表示层的学习是可以共享的，如multi task learning中的思路。所以在我们的方法里，共享的信息为底层参数（如图像领域的前几个卷积层，NLP中的embedding）， 这些底层参数能一定程度上反应了对输入信息的基本刻画。

整个训练过程，网络的loss如下：

![undefined](https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution/Rocket/docs/loss.png)

LB和LB均为CTR任务的交叉熵Loss，OB为Block梯度的Boost Net输出层向量,  OL为Light Net输出层向量。 

#### Co-Training

两个网络一起训练，从而booster net的$z(x)$会全程监督小网络$l(x)$的学习，一定程度上，booster net指导了light net整个求解过程，这与一般的teacher-student 范式下，学习好大模型，用大模型固定的输出作为soft target来监督小网络的学习有着明显区别。

#### Gradient Block

由于booster net有更多的参数，有更强的拟合能力，我们需要给他更大的自由度来学习，尽量减少小网络对他的拖累，我们提出了gradient block的技术，该技术的目的是，在第三项hint loss进行梯度回传时，我们固定booster net独有的参数$W_B$不更新，让该时刻，大网络前向传递得到的$z(x)$，来监督小网络的学习，从而使得小网络向大网络靠近。在预测阶段，抛弃booster net独有的部分，剩下的light net独自用于推断。整个过程就像火箭发射，在开始阶段，助推器（booster）载着卫星（light net）共同前进，助推器（booster）提供动力，推进卫星（light net）的前行，第二阶段，助推器(booster)被丢弃，只剩轻巧的卫星（light net）独自前行。所以，我们把我们的方法叫做Rocket Launching。

## Rocket训练流程
* 详见script/README.md

#### Cite Bib

```
@inproceedings{zhou2017Rocket,
  title={Rocket Launching: A unified and effecient framework for training well-behaved light net},
  author={Zhou, Guorui and Fan, Ying and Cui, Runpeng and Bian, Weijie and Zhu, Xiaoqiang and Kun, Gai},
  booktitle={Proceedings of the 32nd AAAI Conference on Artificial Intelligence},
  year={2018},
  organization={AAAI}
}
```
