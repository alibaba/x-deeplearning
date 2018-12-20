# XDL提供了基于Docker的，通过YARN进行调度的分布式提交工具

## 集群部署
* Hadoop部署   
提交工具基于HADOOP 3.1.0及更新版本，[HADOOP部署官方文档](https://hadoop.apache.org/docs/r3.1.0/hadoop-project-dist/hadoop-common/ClusterSetup.html)   

* GPU绑核的支持(可选)    
基于官方版本YARN的GPU调度，通常对调度到机器的GPU使用不做限制，而类似TF对GPU的使用策略是默认占用全部显存，所以多个进程共享同一个GPU会导致显存不足等问题。      
我们提供了带GPU隔离的HADOOP_PATCH（通过docker环境变量 CUDA_VISIBLE_DEVICES 实现）

* 安装Docker   
Nodemanager所在的机器需要安装Docker和对应的镜像 [参考](https://github.com/alibaba/x-deeplearning/blob/master/xdl/docs/cluster_deploy.md) 

## 分布式任务提交工具xdl_submit

* 安装
在已安装好的HADOOP集群中任意选择一台机器，我们称之为gateway，在上面安装分布式提交工具xdl_submit.py

安装步骤如下：

```
  cd XDL/distributed
  sudo sh install_xdl_submit.sh
```

安装完成之后，可以用如下命令进行分布式任务的提交：

```
  xdl_submit.py --config=xdl_test.json
```
