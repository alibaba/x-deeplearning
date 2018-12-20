#!/bin/bash

echo '## tree_init'
xdl_submit.py --config config.tree_init.json
echo '## finish tree_init'

echo '## train'
ret=`tail -3 train.py |grep "#train(is_training=True)" |wc -l`
if [ $ret -eq 1 ]; then
    echo "Check is_training failed!"
    tail -3 train.py
    exit -1
fi
hadoop fs -rm -f -r hdfs://your/hdfs/path/checkpoint
xdl_submit.py --config config.train.json
echo '## finish train'

echo '## tree_cluster'
xdl_submit.py --config config.tree_cluster.json
echo '## finish tree_cluster'

echo '## train after tree_cluster'
ret=`tail -3 train.py |grep "#train(is_training=True)" |wc -l`
if [ $ret -eq 1 ]; then
    echo "Check is_training failed!"
    tail -3 train.py
    exit -1
fi
hadoop fs -rm -f -r hdfs://your/hdfs/path/checkpoint
xdl_submit.py --config config.train.json
echo '## finish train after tree_cluster'

echo '## finish all'

