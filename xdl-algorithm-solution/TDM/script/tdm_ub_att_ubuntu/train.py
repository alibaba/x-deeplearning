# Copyright (C) 2016-2018 Alibaba Group Holding Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*-coding:utf-8 -*-
import xdl
import mxnet as mx

import ctypes
import json
import numpy as np
import os
import struct
import subprocess
import time

from tdm_layer_master import mx_dnn_layer, FullyConnected3D

'''
  python train.py --run_mode=local --config=config.train.json
'''

config = json.load(open('data/tdm.json', 'r'))
def conf(key):
    return config[key].encode('utf8')
def intconf(key):
    return config[key]

def shell_cmd(command):
    print(command)
    subprocess.call(command, shell=True)
    return
    try:
        retcode = subprocess.call(command, shell=True)
    except:
        retcode = -1
    if retcode != 0:
        raise RuntimeError("shell cmd {command} failed".format(command=command))

def init():
    if intconf('use_hdfs_tree') != 0:
        shell_cmd("mv %s %s.bak" % (conf('tree_pb'), conf('tree_pb')))
        shell_cmd("hadoop fs -get %s/%s/%s %s" %(conf('upload_url'), conf('data_dir'), conf('tree_filename'), conf('tree_pb')))

    _LIB_NAME = 'libselector.so'
    ctypes.CDLL(_LIB_NAME, ctypes.RTLD_GLOBAL)

    from store import Store
    from dist_tree import DistTree
    #s = Store('size=10000000')
    s = Store(conf('tree_store_config'))
    s.load(conf('tree_pb'))
    tree = DistTree()
    tree.set_store(s.get_handle())
    tree.load()

def train(is_training=True):
    #np.set_printoptions(threshold='nan')
    if is_training or xdl.get_task_index() == 0:
        init()
    else:
        return

    file_type = xdl.parsers.txt
    if is_training:
        data_io = xdl.DataIO("tdm", file_type=file_type, fs_type=xdl.fs.hdfs,
                             namenode="hdfs://your/namenode/hdfs/path:9000", enable_state=False)

        feature_count = 69
        for i in xrange(1, feature_count + 1):
            data_io.feature(name=("item_%s" % i), type=xdl.features.sparse, table=1)
        data_io.feature(name="unit_id_expand", type=xdl.features.sparse, table=0)

        data_io.batch_size(intconf('train_batch_size'))
        data_io.epochs(intconf('train_epochs'))
        data_io.threads(intconf('train_threads'))
        tdmop_layer_counts = [int(i) for i in conf('tdmop_layer_counts').split(',')]
        data_io.sgroup_queue_capacity(intconf('train_threads')*intconf('train_batch_size')/sum(tdmop_layer_counts))
        data_io.label_count(2)
        base_path = '%s/%s/' % (conf('upload_url'), conf('data_dir'))
        data = base_path + conf('train_sample') + '_' + r'[\d]+'
        sharding = xdl.DataSharding(data_io.fs())
        sharding.add_path(data)
        paths = sharding.partition(rank=xdl.get_task_index(), size=xdl.get_task_num())
        print 'train: sharding.partition() =', paths
        data_io.add_path(paths)
        iop = xdl.GetIOP("TDMOP")
    else:
        data_io = xdl.DataIO("tdm", file_type=file_type, fs_type=xdl.fs.hdfs,
                             namenode="hdfs://your/namenode/hdfs/path:9000", enable_state=False)

        feature_count = 69
        for i in xrange(1, feature_count + 1):
            data_io.feature(name=("item_%s" % i), type=xdl.features.sparse, table=1)
        data_io.feature(name="unit_id_expand", type=xdl.features.sparse, table=0)

        data_io.batch_size(intconf('predict_batch_size'))
        data_io.epochs(intconf('predict_epochs'))
        data_io.threads(intconf('predict_threads'))
        data_io.sgroup_queue_capacity(intconf('predict_io_pause_num') *
                                      max(int(conf('pr_test_each_layer_retrieve_num')), int(conf('pr_test_final_layer_retrieve_num'))) * 2)
        data_io.label_count(2)
        base_path = '%s/%s/' % (conf('upload_url'), conf('data_dir'))
        data = base_path + conf('test_sample')
        data_io.add_path(data)
        print 'predict: add_path =', data
        iop = xdl.GetIOP("TDMPREDICTOP")
        #data_io.finish_delay(True)
    assert iop is not None
    key_value = {}
    key_value["key"] = "value"
    key_value["debug"] = conf('tdmop_debug')
    key_value["layer_counts"] = conf('tdmop_layer_counts')
    key_value["pr_test_each_layer_retrieve_num"] = conf("pr_test_each_layer_retrieve_num")
    key_value["pr_test_final_layer_retrieve_num"] = conf("pr_test_final_layer_retrieve_num")
    iop.init(key_value)
    data_io.add_op(iop)
    data_io.split_group(False)
    if not is_training:
        data_io.keep_sample(True)
        data_io.pause(intconf('predict_io_pause_num'), True)
    data_io.startup()

    if not is_training:
        if xdl.get_task_index() == 0:
            saver = xdl.Saver()
            saver.restore(conf('saver_ckpt'))

    batch = data_io.read()

    emb_combiner = 'mean'    # mean | sum
    ind = batch["indicators"][0]
    ids = batch["_ids"][0]
    emb = []
    emb_dim = 24
    if is_training:
        feature_add_probability = 1.
    else:
        feature_add_probability = 0.
    import xdl.python.sparse_engine.embedding as embedding
    emb_name = "item_emb"
    for i in xrange(1, feature_count + 1):
        #emb_name = "item_%s_emb" % i
        eb = xdl.embedding(emb_name, batch["item_%s" % i], xdl.Normal(stddev=0.001), emb_dim, 50000, emb_combiner, vtype="hash", feature_add_probability=feature_add_probability)
        with xdl.device('GPU'):
            eb_take = xdl.take_op(eb, batch["indicators"][0])
        eb_take.set_shape(eb.shape)
        emb.append(eb_take)
    #emb_name = "unit_id_expand_emb"
    unit_id_expand_emb = xdl.embedding(emb_name, batch["unit_id_expand"], xdl.Normal(stddev=0.001), emb_dim, 50000, emb_combiner, vtype="hash", feature_add_probability=feature_add_probability)

    @xdl.mxnet_wrapper(is_training=is_training, device_type='gpu')
    def dnn_model_define(user_input, indicator, unit_id_emb, label, bs, eb_dim, fea_groups, active_op='prelu', use_batch_norm=True):
        # 把用户输入按fea_groups划分窗口，窗口内做avg pooling
        fea_groups = [int(s) for s in fea_groups.split(',')]
        total_group_length = np.sum(np.array(fea_groups))
        print "fea_groups", fea_groups, "total_group_length", total_group_length, "eb_dim", eb_dim
        user_input_before_reshape = mx.sym.concat(*user_input)
        user_input = mx.sym.reshape(user_input_before_reshape, shape=(-1, total_group_length, eb_dim))
    
        layer_data = []
        # start att
        att_user_input = mx.sym.reshape(user_input, (bs, total_group_length, eb_dim))
        att_node_input = mx.sym.reshape(unit_id_emb, (bs, 1, eb_dim))
        att_node_input = mx.sym.broadcast_to(data=att_node_input, shape=(0, total_group_length, 0))
        att_din = mx.sym.concat(att_user_input, att_user_input * att_node_input, att_node_input, dim=2)

        att_active_op = 'prelu'
        att_layer_arr = []
        att_layer1 = FullyConnected3D(3*eb_dim, 36, active_op=att_active_op, version=1, batch_size=bs)
        att_layer_arr.append(att_layer1)
        att_layer2 = FullyConnected3D(36, 1, active_op=att_active_op, version=2, batch_size=bs)
        att_layer_arr.append(att_layer2)

        layer_data.append(att_din)
        for layer in att_layer_arr:
            layer_data.append(layer.call(layer_data[-1]))
        att_dout = layer_data[-1]
        att_dout = mx.sym.broadcast_to(data=att_dout, shape=(0, 0, eb_dim))

        user_input = mx.sym.reshape(user_input, shape=(bs, -1, eb_dim))
        user_input = user_input * att_dout
        # end att

        idx = 0
        for group_length in fea_groups:
            block_before_sum = mx.sym.slice_axis(user_input, axis=1, begin=idx, end=idx+group_length)
            block = mx.sym.sum_axis(block_before_sum, axis=1) / group_length
            if idx == 0:
                grouped_user_input = block
            else:
                grouped_user_input = mx.sym.concat(grouped_user_input, block, dim=1)
            idx += group_length
    
        indicator = mx.symbol.BlockGrad(indicator)
        label = mx.symbol.BlockGrad(label)
        # 按indicator来扩展user fea，然后过网络
        #grouped_user_input_after_take = mx.symbol.take(grouped_user_input, indicator)
        grouped_user_input_after_take = grouped_user_input
        din = mx.symbol.concat(*[grouped_user_input_after_take, unit_id_emb], dim=1)
    
        net_version = "d"
        layer_arr = []
        layer1 = mx_dnn_layer(11 * eb_dim, 128, active_op=active_op, use_batch_norm=use_batch_norm, version="%d_%s" % (1, net_version))
        layer_arr.append(layer1)
        layer2 = mx_dnn_layer(128, 64, active_op=active_op, use_batch_norm=use_batch_norm, version="%d_%s" % (2, net_version))
        layer_arr.append(layer2)
        layer3 = mx_dnn_layer(64, 32, active_op=active_op, use_batch_norm=use_batch_norm, version="%d_%s" % (3, net_version))
        layer_arr.append(layer3)
        layer4 = mx_dnn_layer(32, 2, active_op='', use_batch_norm=False, version="%d_%s" % (4, net_version))
        layer_arr.append(layer4)
        #layer_data = [din]
        layer_data.append(din)
        for layer in layer_arr:
            layer_data.append(layer.call(layer_data[-1]))
        dout = layer_data[-1]
    
        # 正常label两列加和必为1，补全的label为0，故减一之后即可得到-1，作为ignore label
        ph_label_sum = mx.sym.sum(label, axis=1)
        ph_label_ignore = ph_label_sum - 1
        ph_label_ignore = mx.sym.reshape(ph_label_ignore, shape=(-1, 1))
        ph_label_click = mx.sym.slice_axis(label, axis=1, begin=1, end=2)
        ph_label_click = ph_label_click + ph_label_ignore
        ph_label_click = mx.sym.reshape(ph_label_click, shape=(bs, ))
    
        prop = mx.symbol.SoftmaxOutput(data=dout, label=ph_label_click, grad_scale=1.0, use_ignore=True, normalization='valid')
        origin_loss = mx.sym.log(prop) * label
        ph_label_sum = mx.sym.reshape(ph_label_sum, shape=(bs, 1))
        origin_loss = mx.sym.broadcast_mul(origin_loss, ph_label_sum)
        loss = - mx.symbol.sum(origin_loss) / mx.sym.sum(ph_label_sum)
        return prop, loss

    re = dnn_model_define(emb, batch["indicators"][0], unit_id_expand_emb, batch["label"], data_io._batch_size, emb_dim, '20,20,10,10,2,2,2,1,1,1')
    prop = re[0]
    loss = re[1]

    if is_training:
        train_op = xdl.Adam(learning_rate=intconf('learning_rate'), lr_decay=False).optimize()
        #train_op = xdl.SGD(0.1).optimize()
        #fc_1_weight_grad = xdl.get_gradient("fc_w_1_d")
        #fc_1_bias_grad = xdl.get_gradient("fc_b_1_d")
    else:
        fin = data_io.set_prop(prop=prop)

    hooks = []
    if is_training:
        if conf("train_mode") == "sync":
            hooks.append(xdl.SyncRunHook(xdl.get_task_index(), xdl.get_task_num()))
        if xdl.get_task_index() == 0:
            ckpt_hook = xdl.CheckpointHook(intconf('save_checkpoint_interval'))
            hooks.append(ckpt_hook)
        log_hook = xdl.LoggerHook([loss], "#### loss:{0}")
    else:
        log_hook = xdl.LoggerHook([loss], "#### loss:{0}")
    hooks.append(log_hook)

    from xdl.python.training.training_utils import get_global_step
    global_step = get_global_step()

    sess = xdl.TrainSession(hooks)

    elapsed_time = 0.
    statis_begin_loop = 200
    loop_num = 0
    while not sess.should_stop():
        print ">>>>>>>>>>>> %d >>>>>>>>>>>" % loop_num
        begin_time = time.time()
        for itr in xrange(200):
            if is_training:
                result = sess.run([train_op, xdl.get_collection(xdl.UPDATE_OPS)])
                #result = sess.run([train_op, xdl.get_collection(xdl.UPDATE_OPS), unit_id_expand_emb])
            else:
                result = sess.run([loss, fin, global_step.value])
                #result = sess.run([loss, fin, ids, global_step.value])
            if result is None:
                print "result is None, finished success."
                break
            if not is_training:
                print "global_step =", result[-1]
                #print "batch['_ids'] =", result[-2]
            #else:
            #   print "unit_id_expand_emb = { mean =", result[-1].mean(), ", std =", result[-1].std(), "}"
            loop_num += 1
        if loop_num > statis_begin_loop:
            elapsed_time += time.time() - begin_time
            #print 'batch_size = %d, qps = %f batch/s' % (data_io._batch_size, (loop_num - statis_begin_loop) / elapsed_time)

    if is_training:
        xdl.execute(xdl.ps_synchronize_leave_op(np.array(xdl.get_task_index(), dtype=np.int32)))
        if xdl.get_task_index() == 0:
            print 'start put item_emb'
            def _string_to_int8(src):
                return np.array([ord(ch) for ch in src], dtype=np.int8)
            from xdl.python.utils.config import get_ckpt_dir
            output_dir = conf('model_url')
            op = xdl.ps_convert_ckpt_variable_op(checkpoint_dir=_string_to_int8(get_ckpt_dir()), 
                                                 output_dir=_string_to_int8(output_dir), 
                                                 variables=_string_to_int8("item_emb"))
            xdl.execute(op)
            shell_cmd("rm -f data/item_emb")
            shell_cmd("hadoop fs -get %s/item_emb data/item_emb" % output_dir)
            shell_cmd("sed -i 's/..//' data/item_emb")
            shell_cmd("hadoop fs -put -f data/item_emb %s" % output_dir)
            print 'finish put item_emb'
        #print 'before worker barrier'
        #xdl.execute(xdl.worker_barrier_op(np.array(xdl.get_task_index(), dtype=np.int32), np.array(xdl.get_task_num(), dtype=np.int32)))
        #print 'after worker barrier'

train(is_training=True)
#train(is_training=False)

