#!/usr/bin/sh

export HADOOP_CONF_DIR=`pwd`/etc/hadoop/
export HADOOP_HOME=`pwd`
export HADOOP_HDFS_HOME=`pwd`
sbin/stop-dfs.sh
rm -rf /tmp/xdl_test/hadoop
