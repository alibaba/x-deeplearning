# 高性能的Blaze模型http协议服务器

## 如何配置
此服务器的配置文件格式在serving/frame/predict.proto中定义。
配置文件应该遵从protobuf的文本格式[TextFormat](https://developers.google.com/protocol-buffers/docs/overview#whynotxml)

这个配置文件会告诉服务器要去加载哪些模型，以及这些模型的地址。
对于每一个模型来说，你需要指定它的名字，及model\_version，以及相应的模型文件，包括dense部分和sparse部分。
其中dense部分是必须的，而sparse部分是可选的。

我们在server/config中给出了一个配置文件的例子。
这个配置文件包含了四个模型。分别为:

  * TDM-DNN: TDM dnn模型
  * TDM-ATT: TDM attention模型
  * DIN: DIN模型
  * GwEN: GwEN模型

## 协议
客户端与服务端之间的协议同样定义在serving/frame/predict.proto中，message Request和message Response。
这里要注意的是，不同于配置文件采用的TextFormat，客户端与服务端之间通过http协议发送的消息是protobuf的JsonFormat。
此举是为了提高此示例的可读性，如果你想得到更高的性能，可以将其改成protobuf的BinaryFormat。

## 运行
安装完blaze之后，执行如下两步启动打分服务

### 模型转换
```bash
cd /opt/blaze/tools/example_model
sudo sh build_all_model.sh
```

### 服务启动
```bash
/opt/blaze/bin/server /opt/blaze/conf/config
```
服务端端口为8080，请确保没有端口冲突。

### 客户端请求
serving/client.py是一个简单的客户端示例，它会发送条对dnn3模型的打分请求。
```bash
python /opt/blaze/bin/client.py
```

期望的输出如下
```text
{"outputList":[{"name":"softmaxoutput0","shape":[1,2],"value":[0.011014563,0.988985419]}]}
```

由于冷启动的现象，刚开始的几次请求可能会有较长的response time。
当服务端预热好之后，性能会大幅提升并稳定下来。

## 优化方案
这个服务器只是一个简单的demo，可能无法满足对性能要求极高的生产环境。
下面给出一些可行的优化方案。
1. 使用别的高性能的rpc框架而不是HTTP服务，毕竟HTTP协议栈耗时在这种场景下并不是可忽略的。
2. 客户端与服务端之间的通讯使用protocol的BinaryFormat而不是JsonFormat，这将大大提高序列化/反序列化的效率。
3. 在ModelManager类中创建blaze::predictor的对象池，而不是在每个session中重复创建和销毁。
4. 使用memcpy来将Blaze output中的数据拷贝到返回消息的protobuf中，而不是像此示例中一个一个赋值。（然而这种做法依赖与protobuf的实现，可能造成兼容性问题）
