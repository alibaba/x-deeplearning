## [ESMM开源数据集](https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408)
## 使用该工具将原始数据格式转换为XDL格式
### 1.下载开源数据集
> 下载训练样本: sample_train.tar.gz，解压缩得到sample_skeleton_train.csv和common_features_train.csv
>
> 下载测试样本: sample_test.tar.gz，解压缩得到sample_skeleton_test.csv和common_features_test.csv

### 2.编译并运行转换程序
```
mkdir build && cd build
cmake .. && make -j32
./sample_generator sample_skeleton_train.csv common_features_train.csv output_prefix worker_num file_num
```
该工具将生成*file_num*个样本文件，文件名为*output_prefix.0* ~ *output_prefix.${n-1}*
