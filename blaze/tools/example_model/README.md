## 模型列表

  * din
  * gwen
  * tdm\_dnn
  * tdm\_tdm

  每个模型文件夹下的model目录下存放可服务的模型文件。

## 模型文件说明

  * graph\_ulf.txt是人工定义的ULF格式在线模型结构, ad\_embedding.conf和user\_embedding.conf是ULF中embedding\_layer的参数配置(___如果用户调整模型结构，需按照ULF的格式说明修改graph\_ulf.txt___)

  * dense.txt  graph.txt  item\_emb  sparse.txt等其它文件是XDL生产出的模型结构.

