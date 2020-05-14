# 使用KubeDL Operator运行XDL

这个教程将说明，如何在Kubernetes中运行分布式XDL训练任务。

## 依赖

在开始教程前，我们需要有一个完整的Kubernetes集群，同时需要在Kubernetes集群中安装上[KubeDL Operator](https://github.com/alibaba/kubedl#getting-started)，并且[开启XDLJob支持](https://github.com/alibaba/kubedl#optional-enable-workload-kind-selectively)。

## 安装ZooKeeper服务

XDL强依赖ZooKeeper进行服务发现，所以我们需要先安装ZooKeeper服务。

下面的命令将按照一个单节点的ZooKeeper服务。

```bash
kubectl apply -f https://raw.githubusercontent.com/alibaba/x-deeplearning/master/xdl/docs/tutorial/v1/xdl-zk.yaml
```

对于生产环境，可以遵循Kubernetes[官方文档](https://kubernetes.io/docs/tutorials/stateful-application/zookeeper/)来安装一个3节点ZooKeeper集群。

## 运行XDLJob训练

我们需要生成一个XDLJob的Yaml部署文件，并且在里面设置好ZooKeeper服务地址。对于所有XDLJob的容器，KubeDL Operator会在容器的环境变量中增加```TASK_NAME``` 和 ```TASK_INDEX```来区分每个容器的身份。同时KubeDL Operator会修改环境变量```ZK_ADDR```来加上XDLJob的UUID。

下面的命令就是运行一个XDLJob。

```bash
kubectl apply -f https://raw.githubusercontent.com/alibaba/x-deeplearning/master/xdl/docs/tutorial/v1/xdl-job.yaml
```

## 查看XDLJob运行情况

查看XDLJob是否正常拉起，所有的Pod是否正常运行。

```bash
kubectl get xdljob

NAME                STATE     AGE   FINISHED-TTL   MAX-LIFETIME
xdl-mnist-example   Running   70s   3600

kubectl get po

NAME                            READY   STATUS    RESTARTS   AGE
xdl-mnist-example-ps-0          1/1     Running   0          116s
xdl-mnist-example-scheduler-0   1/1     Running   0          116s
xdl-mnist-example-worker-0      1/1     Running   0          116s
xdl-mnist-example-worker-1      1/1     Running   0          116s
zk-c5cc46c8d-s6bkc              1/1     Running   0          2m26s
```
