## 简介
Blaze是一个面向广告/搜索/推荐场景的高性能深度学习推理引擎。

### 前置依赖
  * gcc >= 4.8.5
  * cuda >= 8.0
  * cudnn >= 6.0
  * mkl
  * cmake >= 2.8
  * python >= 2.7 (推荐使用Anaconda)

MKL 可以在 https://software.intel.com/en-us/mkl/choose-download 注册并下载，然后安装到本地目录。推荐下载 2018 Update 3，此为经过验证通过的版本。

### 编译安装

  * 编译仅支持 CPU 的版本, 安装路径可通过CMAKE\_INSTALL\_PREFIX指定，默认安装路径为: /opt/blaze/

```bash
  $ mkdir build
  $ cd build
  $ cmake ../ -DUSE_CUDA=0 -DUSE_MKL=1 -DMKL_PATH=<mkl-root-dir> -DSYMBOL_EXPORT_CTL=0
  $ make -j 8
  $ sudo make install
  $ cd ../binding/python
  $ sudo python setup.py install
```

其中，`<mkl-root-dir>` 是 MKL 的安装绝对路径，且需要指定根目录。例如，MKL 安装到 `/home/user/intel` 下，则 `mkl-root-dir` 为 `/home/user/intel`。当TDM-Serving使用，采用-DSYMBOL_EXPORT_CTL=1

  * 编译支持 CPU 和 GPU 的版本, 安装路径可通过CMAKE\_INSTALL\_PREFIX指定, 默认安装路径为: /opt/blaze/

```
  $ mkdir build
  $ cd build
  $ cmake ../ -DUSE_CUDA=1 -DCUDA_TOOLKIT_ROOT_DIR=<cuda-root-dir> \
          -DCUDNN_ROOT_DIR=<cudnn-root-dir> \
          -DUSE_MKL=1 -DMKL_PATH=<mkl-root-dir> -DSYMBOL_EXPORT_CTL=0
  $ make -j 8
  $ sudo make install
  $ cd ../binding/python
  $ sudo python setup.py install
```

其中，`<cuda-root-dir>` 是 CUDA 的安装绝对路径，例如 `/usr/local/cuda-8.0`，`<cudnn-root-dir>` 是 cuDNN 的安装路径，例如 `/usr/local`。`<mkl-root-dir>` 与编译 CPU 的版本时设定方法相同。当TDM-Serving使用，采用-DSYMBOL_EXPORT_CTL=1

## 内部模型格式
  
   为提供高性能的在线模型服务，Blaze自定义了面向广告场景的模型结构描述ULF, 详见[__ULF格式说明__](blaze/model_importer/ulf.md), 需要用户手动自定义模型结构。

## 模型服务构建
   
### 模型转换
  
   XDL产出的模型结构，需要进行模型转换才能被Blaze加载并提供服务，详细介绍见[__转换工具__](tools/)。

### 服务接入

   我们提供一个使用blaze搭建的简单的模型服务样例，服务端接收模型输入数据并返回模型打分结果，详细介绍见[__模型服务__](serving/README.md)。
