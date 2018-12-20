#!/usr/bin/sh

export HADOOP_CONF_DIR=`pwd`/etc/hadoop/
export HADOOP_HOME=`pwd`
export HADOOP_HDFS_HOME=`pwd`
sbin/stop-dfs.sh
rm -rf /tmp/xdl_test/hadoop
echo Y|bin/hdfs namenode -format
echo Y|bin/hdfs datanode -format
sbin/start-dfs.sh
bin/hdfs dfs -put ../../test_data hdfs://127.0.0.1:9090/
