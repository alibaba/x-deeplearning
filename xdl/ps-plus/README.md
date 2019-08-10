/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ps-plus使用

ps-plus是底层的ps组件，可单独介入到其他的业务系统中。简单实用如下：

##编译

```
git clone --recursive https://github.com/alibaba/x-deeplearning.git
cd x-deeplearning/xdl
mkdir build && cd build
export CC=/usr/bin/gcc-5 && export CXX=/usr/bin/g++-5
cmake .. -DTF_BACKEND=1
make -j8 
```

##运行Demo
make完成后，在xdl/build/ps-plus目录下，会生成ps-plus相关的lib以及test binary。目前提供一组Demo，可使用下面命令启动：

```
1. 首先在本地创建一个zk节点，scheduler需要。

apt-get install zookeeper
/usr/share/zookeeper/bin/zkServer.sh start
/usr/share/zookeeper/bin/zkCli.sh create /scheduler "scheduler"
/usr/share/zookeeper/bin/zkCli.sh get /scheduler 
```

```
2. 启动scheduler
./ps -r scheduler -sp zfs://localhost:2181/scheduler  -cp . -bc false -p 8801 -sn 1 -snet 10000 -smem 10000 -sqps 100000 
```

```
3. 启动一个server
./ps -r server -p 8802 -si 0 -sp zfs://localhost:2181/scheduler -bc false
```

```
4. 启动一个clie	nt
./tool -v val -sn 1 -sp zfs://localhost:2181/scheduler -a testxxx
```

参数解释：

```
Demo文件路径
scheduler和server文件：
x-deeplearning/xdl/ps-plus/ps-plus/main/main.cc
client文件：
x-deeplearning/xdl/ps-plus/ps-plus/tool/client_tool.cpp

--r：表示启动的角色是server还是scheduler
-sp：表示scheduler_kv_path
-sn：表示需要启动几个server
-cp：checkpoint_path
-snet：server端流量限制
-smem：server端memory限制
-sqps：server端QPS限制
-bc：是否线程绑核(测试不建议绑核)
-si：这个只有server需要的参数，表示当前server编号，从0开始
```

目前Demo代码，默认会使用docker中分配的所有的cpu core，后期需要加上core数量参数指定。

