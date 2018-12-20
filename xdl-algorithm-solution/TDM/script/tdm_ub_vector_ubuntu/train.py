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
        shell_cmd("hadoop fs -get %s/%s/%s %s" % (conf('upload_url'), conf('data_dir'), conf('tree_filename'), conf('tree_pb')))

    _LIB_NAME = 'libselector.so'
    ctypes.CDLL(_LIB_NAME, ctypes.RTLD_GLOBAL)

    from store import Store
    from dist_tree import DistTree
    s = Store(conf('tree_store_config'))
    s.load(conf('tree_pb'))
    tree = DistTree()
    tree.set_store(s.get_handle())
    tree.load()


def train(is_training=True):
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
        data_io.feature(name="test_unit_id", type=xdl.features.sparse, table=1)

        data_io.batch_size(intconf('predict_batch_size'))
        data_io.epochs(intconf('predict_epochs'))
        data_io.threads(intconf('predict_threads'))
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
    key_value["start_sample_layer"] = "22"
    key_value["pr_test_each_layer_retrieve_num"] = "400"
    key_value["pr_test_final_layer_retrieve_num"] = "200"
    if not is_training:
        key_value["expand_mode"] = "vector"
    iop.init(key_value)
    data_io.add_op(iop)
    data_io.split_group(False)
    data_io.startup()

    if not is_training:
        if xdl.get_task_index() == 0:
            saver = xdl.Saver()
            saver.restore(conf('saver_ckpt'))

    batch = data_io.read()

    emb_combiner = 'mean'    # mean | sum
    if not is_training:
        gt_ids = batch["_ids"][-1]
        gt_segments = batch["_segments"][-1]
    emb = []
    emb_dim = 24
    if is_training:
        feature_add_probability = 1.
    else:
        feature_add_probability = 0.
    import xdl.python.sparse_engine.embedding as embedding
    emb_name = "item_emb"
    for i in xrange(1, feature_count + 1):
        eb = xdl.embedding(emb_name, batch["item_%s" % i], xdl.Normal(stddev=0.001), emb_dim, 50000, emb_combiner, vtype="hash", feature_add_probability=feature_add_probability)
        with xdl.device('GPU'):
            eb_take = xdl.take_op(eb, batch["indicators"][0])
        eb_take.set_shape(eb.shape)
        emb.append(eb_take)
    unit_id_expand_emb = xdl.embedding(emb_name, batch["unit_id_expand"], xdl.Normal(stddev=0.001), emb_dim, 50000, emb_combiner, vtype="hash", feature_add_probability=feature_add_probability)

    @xdl.mxnet_wrapper(is_training=is_training, device_type='gpu')
    def dnn_model_define(user_input, indicator, unit_id_emb, label, bs, eb_dim, sample_num, fea_groups, active_op='prelu', use_batch_norm=True):
        # 把用户输入按fea_groups划分窗口，窗口内做avg pooling
        fea_groups = [int(s) for s in fea_groups.split(',')]
        total_group_length = np.sum(np.array(fea_groups))
        print "fea_groups", fea_groups, "total_group_length", total_group_length, "eb_dim", eb_dim
        user_input_before_reshape = mx.sym.concat(*user_input)
        user_input = mx.sym.reshape(user_input_before_reshape, shape=(-1, total_group_length, eb_dim))

        idx = 0
        for group_length in fea_groups:
            block_before_sum = mx.sym.slice_axis(user_input, axis=1, begin=idx, end=idx + group_length)
            block = mx.sym.sum_axis(block_before_sum, axis=1) / group_length
            if idx == 0:
                grouped_user_input = block
            else:
                grouped_user_input = mx.sym.concat(grouped_user_input, block, dim=1)
            idx += group_length

        indicator = mx.symbol.BlockGrad(indicator)
        label = mx.symbol.BlockGrad(label)
        grouped_user_input_after_take = grouped_user_input

        net_version = "e"
        layer_arr = []
        layer1 = mx_dnn_layer(10 * eb_dim, 128, active_op=active_op, use_batch_norm=use_batch_norm, version="%d_%s" % (1, net_version))
        layer_arr.append(layer1)
        layer2 = mx_dnn_layer(128, 64, active_op=active_op, use_batch_norm=use_batch_norm, version="%d_%s" % (2, net_version))
        layer_arr.append(layer2)
        layer3 = mx_dnn_layer(64, 24, active_op='', use_batch_norm=False, version="%d_%s" % (3, net_version))
        layer_arr.append(layer3)

        layer_data = [grouped_user_input_after_take]
        for layer in layer_arr:
            layer_data.append(layer.call(layer_data[-1]))
        dout = layer_data[-1]

        inner_product = mx.sym.sum(dout * unit_id_emb, axis=1)

        softmax_input = mx.sym.Reshape(inner_product,
                                       shape=(
                                           bs / sample_num,
                                           sample_num
                                       )
                                       )

        # 用正例的label减1作为softmax的label
        ph_label_click = mx.sym.slice_axis(label, axis=1, begin=1, end=2)
        ph_label_click = mx.sym.reshape(ph_label_click, shape=(bs / sample_num, sample_num)) - 1
        ph_label_click = mx.sym.slice_axis(ph_label_click, axis=1, begin=0, end=1)
        ph_label_click = mx.sym.reshape(ph_label_click, shape=(bs / sample_num, ))

        prop = mx.symbol.SoftmaxOutput(data=softmax_input, label=ph_label_click, normalization='valid', use_ignore=True)

        positive_prop = mx.sym.slice_axis(prop, axis=1, begin=0, end=1)
        positive_prop = mx.sym.reshape(positive_prop,
                                       shape=(bs / sample_num, )
                                       )

        # 实际的有效样本数量是(bs/sample_num)减去需要ignore的label数量
        loss = -mx.sym.sum(mx.symbol.log(positive_prop)) / (bs / sample_num + mx.sym.sum(ph_label_click))

        user_vector = mx.sym.reshape(dout, shape=(bs / sample_num, sample_num, eb_dim))
        user_vector = mx.sym.slice_axis(user_vector, axis=1, begin=0, end=1)
        user_vector = mx.sym.reshape(user_vector, shape=(bs / sample_num, eb_dim))

        return prop, loss, mx.sym.BlockGrad(user_vector)

    if is_training:
        re = dnn_model_define(emb, batch["indicators"][0], unit_id_expand_emb, batch["label"], data_io._batch_size, emb_dim, 600, '20,20,10,10,2,2,2,1,1,1')
    else:
        re = dnn_model_define(emb, batch["indicators"][0], unit_id_expand_emb, batch["label"], data_io._batch_size, emb_dim, 1, '20,20,10,10,2,2,2,1,1,1')
    prop = re[0]
    loss = re[1]

    if is_training:
        train_op = xdl.Adam(learning_rate=intconf('learning_rate')).optimize()
    else:
        user_vector = re[2]
 
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

    if not is_training:
        urun_re = iop.urun({"get_level_ids": key_value["start_sample_layer"]})
        item_num = len(urun_re)
        item_ids = np.array([int(iid) for iid in urun_re.keys()], dtype=np.int64).reshape((item_num, 1))
        print 'item_ids shape: '
        print item_ids.shape
        zeros = np.zeros((item_num, 1), dtype=np.int64)
        hash_ids = np.concatenate((zeros, item_ids), axis=1)
        item_embeddings = xdl.execute(xdl.ps_sparse_pull_op(hash_ids, var_name="item_emb", var_type="hash", save_ratio=1.0, otype=xdl.DataType.float))
        item_embeddings = item_embeddings.transpose()
        print 'item_embeddings shape: '
        print item_embeddings.shape

        hit_num_list = []
        precision_list = []
        recall_list = []
        gt_num_list = []
        user_idx = 1

    while not sess.should_stop():
        print ">>>>>>>>>>>> %d >>>>>>>>>>>" % loop_num
        begin_time = time.time()
        for itr in xrange(200):
            if is_training:
                result = sess.run([train_op, xdl.get_collection(xdl.UPDATE_OPS)])
            else:
                result = sess.run([user_vector, global_step.value, gt_ids, gt_segments])
            if result is None:
                print "result is None, finished success."
                break
            if not is_training:
                print "global_step =", result[1]
                batch_uv = result[0]
                batch_gt = result[2]
                batch_seg = result[3]

                batch_uv = batch_uv[0:len(batch_seg)]
                batch_scores = np.matmul(batch_uv, item_embeddings)

                sorted_idx = np.argsort(-batch_scores, axis=1)

                sorted_idx = sorted_idx[:, :int(key_value["pr_test_final_layer_retrieve_num"])]
                gt_id_start_idx = 0
                for i in xrange(len(batch_seg)):
                    pred_set = set(item_ids[sorted_idx[i, :], 0])
                    gt_dict = {}
                    for gt in batch_gt[gt_id_start_idx:batch_seg[i], 1]:
                        if gt in gt_dict:
                            gt_dict[gt] += 1
                        else:
                            gt_dict[gt] = 1

                    test_gt_list = batch_gt[gt_id_start_idx:batch_seg[i], 1].tolist()
                    test_gt_str = ','.join([str(gtid) for gtid in test_gt_list])
                    test_pred_list = item_ids[sorted_idx[i, :], 0].tolist()
                    test_pred_str = ','.join([str(gtid) for gtid in test_pred_list])

                    user_idx += 1

                    gt_set = set(batch_gt[gt_id_start_idx:batch_seg[i], 1])
                    comm_set = gt_set.intersection(pred_set)

                    hit_num = sum([float(gt_dict[item]) if item in gt_dict else 0.0 for item in comm_set])
                    hit_num_list.append(hit_num)

                    if len(pred_set) > 0:
                        precision = hit_num / len(pred_set)
                    else:
                        precision = 0.0

                    if len(gt_dict) > 0:
                        recall = hit_num / (batch_seg[i] - gt_id_start_idx)
                    else:
                        recall = 0.0

                    precision_list.append(precision)
                    recall_list.append(recall)
                    gt_num_list.append(float(batch_seg[i] - gt_id_start_idx))

                    gt_id_start_idx = batch_seg[i]

                print "=================================================="
                print 'predicted user num is: %d' % len(hit_num_list)
                print 'gt num is: %f' % sum(gt_num_list)
                print 'precision: %f' % (sum(precision_list) / len(hit_num_list))
                print 'recall: %f' % (sum(recall_list) / len(hit_num_list))
                print 'global recall: %f' % (sum(hit_num_list) / sum(gt_num_list))
                print "=================================================="

            loop_num += 1
        if loop_num > statis_begin_loop:
            elapsed_time += time.time() - begin_time
            #print 'batch_size = %d, qps = %f batch/s' % (data_io._batch_size, (loop_num - statis_begin_loop) / elapsed_time)

    if not is_training:
        print "=================================================="
        print 'predicted user num is: %d' % len(hit_num_list)
        print 'gt num is: %f' % sum(gt_num_list)
        print 'precision: %f' % (sum(precision_list) / len(hit_num_list))
        print 'recall: %f' % (sum(recall_list) / len(hit_num_list))
        print 'global recall: %f' % (sum(hit_num_list) / sum(gt_num_list))
        print "=================================================="

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
# train(is_training=False)

