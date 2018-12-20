<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# Deep Interest Network

# 算法介绍
## DIN( Deep Interest Network) 
DIN是由阿里妈妈精准定向广告算法团队在KDD2018提出的针对电商场景深入理解用户兴趣的预估模型，也可适用于其他场景。

随着硬件计算能力发展带动深度学习的进步，预估领域的算法也逐渐的从传统的CTR（Click Through-Rate）预估模型迁移到深度CTR预估模型。
这些模型都可以归结为Embedding&MLP的范式：首先通过embedding layer将大规模的稀疏特征投影为低维连续的embedding vector，
然后将这些向量concatate后输入到一个全连接网络中，计算其最终的预估目标。相较于传统的方法，
这样的做法利用了深度学习模型更强的model capacity自动的学习特征之间以及特征与目标的关系，
减少了传统模型需要人工经验和实验验证的特征筛选设计阶段。

后续研究者们提出了一些工作在这个范式基础上利用传统的技术做了一些进一步的改进，然而当我们面对电商的预估场景，这些方法都缺乏对此场景的理解。
在电商场景如果试图做到精准的预估必须充分利用用户的历史行为来理解用户的兴趣。传统的Embedding&MLP模型用一个固定的向量来表达一个用户。
在电商场景，如一个用户来到淘宝，他会同时存在多种不同的兴趣对不同的商品都有潜在的兴趣，这也反应在用户的历史行为里，用户历史上交互的商品也是多种多样的。
因此前面提到的Embedding&MLP模型利用一个固定的向量![img](http://latex.codecogs.com/svg.latex?V_u)去表达用户就会成为模型表达用户多样兴趣的瓶颈。此固定用户向量![img](http://latex.codecogs.com/svg.latex?V_u)的维度限制了整体模型解空间的秩，
而向量的维度受计算力以及泛化性的限制不可能无限制的扩充。为了缓解这一问题，我们提出用一个根据预估目标动态变换的向量来表达用户。实际上当我们要预测一个User ![img](http://latex.codecogs.com/svg.latex?U)
对一个目标Item ![img](http://latex.codecogs.com/svg.latex?I_t) 的点击率，我们可能并不需要![img](http://latex.codecogs.com/svg.latex?V_u)表达用户的所有兴趣，我们只需要表达其和![img](http://latex.codecogs.com/svg.latex?I_t)相关的兴趣。
比如影响一个用户是否点击一件衣服的时候，其对家用电器的兴趣可能并不会影响这一次决定。基于以上我们提出了DIN模型捕捉针对不同商品时用户不同的兴趣状态，并用一个根据不同预估商品目标动态变换的![img](http://latex.codecogs.com/svg.latex?V_u)来表达用户与之相关的兴趣。

DIN的模型结构如下图。DIN通过一个兴趣激活模块(Activation Unit)，用预估目标Candidate ADs的信息去激活用户的历史点击商品，以此提取用户与当前预估目标相关的兴趣。权重高的历史行为表明这部分兴趣和当前广告相关，权重低的则是和广告无关的”兴趣噪声“。我们通过将激活的商品和激活权重相乘，然后累加起来作为当前预估目标ADs相关的兴趣状态表达。
最后我们将这相关的用户兴趣表达、用户静态特征和上下文相关特征，以及ad相关的特征拼接起来，输入到后续的多层DNN网络，最后预测得到用户对当前目标ADs的点击概率。
![undefined](https://cdn.nlark.com/lark/0/2018/png/36154/1541994938186-65f7eebc-b246-4fa1-9d35-5cad9e22a08d.png) 

这里是实际预测过程中DIN中兴趣激活模块根据预估目标对历史行为预测的相关权重，黄色能量条的长度越长表明其激活权重越高，和预估目标更相关。可以看到直观上和此次的预估目标羽绒服相关的商品都获得了相对较高的权重。
![attention_timeline_fix.png](https://cdn.nlark.com/lark/0/2018/png/6098/1542337777101-ea941a64-4f6c-4922-8165-b8bcde577b9a.png) 
## Dice
同时在DIN的论文中，我们还对激活函数做了改进，提出了黑魔法DICE激活函数。

目前应用的比较广的激活函数ReLU和PReLU的计算过程可以描述为：
![CodeCogsEqn-2.svg](https://cdn.nlark.com/lark/0/2018/svg/6098/1542338886126-c86a2268-870f-4f3f-af5b-d2c3a405e769.svg) 

当然也等价于：![CodeCogsEqn-3.svg](https://cdn.nlark.com/lark/0/2018/svg/6098/1542338959431-f13437d4-a951-4d62-9114-d4762b2f1f01.svg) 

我们把![img](http://latex.codecogs.com/svg.latex?P(s)) 称为控制函数，这个控制函数其实就是一个整流器，可以看到无论是PReLU还是ReLU的控制函数
都是一个阶跃函数，其变化点在![img](http://latex.codecogs.com/svg.latex?s=0) 处，意味着面对不同的输入这个变化点是不变的。实际上神经元的输出分布是不同的，
面对不同的数据分布采用同样的策略可能是不合理的，因此在我们提出的Dice中我们改进了这个控制函数，让它根据数据的分布来调整，这里我们选择了统计神经元输出的均值和方差来描述数据的分布：

![CodeCogsEqn-4.svg](https://cdn.nlark.com/lark/0/2018/svg/6098/1542339395366-782ed784-f8ce-42e8-a25c-8d17b4559061.svg) 

![dice_activation.png](https://cdn.nlark.com/lark/0/2018/png/6098/1542339525050-8b030a4a-bd9f-4891-acc1-6ab02192365d.png)

如此，Dice的控制器会根据数据的分布自适应的调整，整体的学习和表达能力都会得到提高。

# DIN训练流程
* 详见script/README.md

## Cite Bib
```
@inproceedings{zhou2018deep,
  title={Deep interest network for click-through rate prediction},
  author={Zhou, Guorui and Zhu, Xiaoqiang and Song, Chenru and Fan, Ying and Zhu, Han and Ma, Xiao and Yan, Yanghui and Jin, Junqi and Li, Han and Gai, Kun},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1059--1068},
  year={2018},
  organization={ACM}
}
```