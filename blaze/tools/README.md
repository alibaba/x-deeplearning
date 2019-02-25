## 模型转换步骤

XDL产出的模型，需要进行模型转换才能提供线上服务.

主要步骤如下:

### 稀疏模型转换命令

```bash
  > python build_qed.py -p xdl/tdm_model/ -i meta -o /tmp/sparse_qed -s 0

       -p XDL产出的稀疏模型路径
       -i XDL产出的稀疏模型的meta文件路径
       -o 模型转换输出文件路径
       -s 输出模型的精度 0/FP32  1/FP16
```

### 稠密模型转换命令
        
```bash
  > python model_converter.py -c xdl/graph.txt -d xdl/dense.txt -o /tmp/model_blaze -b 1

       -c XDL产出的稠密模型结构文件路径
       -d XDL产出的稠密模型参数文件路径
       -o 模型转换输出文件路径
       -b 输出模型格式, 0文本，1二进制，blaze只能加载二进制的模型, 文本用于调试使用
```

### 稠密模型优化命令

```bash
  > python model_optimizer.py -i /tmp/model_blaze -o /tmp/model_blaze_optimized -b 1

       -i 转换后的模型路径
       -o 优化后的模型输出路径
       -b 输出模型格式, 0文本，1二进制
```


___模型转换案例见[example\_model](example_model/README.md)___


