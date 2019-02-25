# 关于Blaze API的示例

## API定义文件所在位置
- C/C++ API 位于 **blaze/api/**
- python API 位于 **binding/blaze/pyblaze**

## API的结构
Blaze的API主要包含了两个类，一个是PredictorManager，另一个是Predictor。
每一个PredictorManager的实例都对应了一个模型model，
而每一个被此实例创造出来的Predictor都对应了一个使用这个模型打分的工作线程。

## 使用此API的过程

1. 初始化PredictorManager并加载模型
2. 为每个工作线程创建一个Predictor实例
3. 为此Predictor指定input的size并输入input数据
4. 在此Predictor上做前向计算
5. 从完成前向计算的Predictor提取输出
6. 释放这个Predictor的实例

## 性能监测
在调用Predictor::Forward()前，使用Predictor::RegisterObservers(observer_names)来注册监测项.
现在支持的监测项被命名为"profile"和"cost"，分别对应了对吞吐和延迟的测量.
在Predictor::Forward()之后调用Predictor::DumpObservers()来获得监测结果.

## 调试Debug
调用Predictor::ListInternalNames()来获得内部tensor的名称, 
包括中间结果（变量）和网络参数（常量）的名称. 
调用Predictor::InternalParam()来获取这些tensor的数据. 
此外,你还可以调用Predictor::InternalShape()来获得这些tensor的形状大小.
[](InternalParam()这个函数名有歧义，我以为是要打印模型参数。是否要修改？)

## 注意事项
在C/C++ API中，PredictorHandle/Predictor对象并不是线程安全的。
因此，用户不应该在同一时刻从不同线程访问同一个PredictorHandle/Predictor实例。
当需要并行打分时，用户应该从PredictorManager中创建出新的Predictor实例。
当此实例被创建出来后，会以指针的形式传递给用户，用户拥有该实例内存的所有权。
结束使用后，务必记得要释放该实例内存以免造成内存泄漏。
