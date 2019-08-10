# Copyright 2018 Alibaba Group. All Rights Reserved.
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

import xdl
import mxnet as mx
import time

import sys
import os
import ctypes
from xdl.python.utils.collections import READER_HOOKS, get_collection

'''
--run_mode=local --ckpt_dir=./ckpt --task_num=1
'''

data_io = xdl.DataIO("tdm", file_type=xdl.parsers.txt, fs_type=xdl.fs.local, enable_state=False)
data_io.epochs(1)
data_io.batch_size(4)
data_io.label_count(1)
#data_io.serialized(True)

embs_len = 4
for i in xrange(1, embs_len + 1):
    name = "item_%d" % i
    data_io.feature(name=name, type=xdl.features.sparse, dim=1)
data_io.add_path('tdm.dat')
data_io.startup()

def train():
    batch = data_io.read()
    print batch 

    embs = list()

    for i in range(1, embs_len + 1):
        name = "item_%d" % i
        emb = xdl.embedding(name, batch[name], xdl.Ones(), 1, 1000, 'sum', vtype='hash')
        embs.append(emb)
        print "emb =", name, ", shape =", emb.shape
    print "origin batch[label].shape =", batch["label"].shape

    loss, prop, label, indicator, din, dout, fc1_weight, fc1_bias, fc2_weight, fc2_bias = model(embs, batch["label"], 4, 7)
    train_op=xdl.SGD(0.5).optimize()

    item1_grad = xdl.get_gradient('item_1')
    item2_grad = xdl.get_gradient('item_2')
    item3_grad = xdl.get_gradient('item_3')
    item4_grad = xdl.get_gradient('item_4')
    fc1_weight_grad = xdl.get_gradient('fc1_weight')
    fc1_bias_grad = xdl.get_gradient('fc1_bias')
    fc2_weight_grad = xdl.get_gradient('fc2_weight')
    fc2_bias_grad = xdl.get_gradient('fc2_bias')

    sess = xdl.TrainSession()
    
    loop_num = 0
    while not sess.should_stop():
        if loop_num == 5:
            break
        print "\n>>>>>>>>>>>> loop_num = %d" % loop_num
        result = sess.run([train_op, loss, prop, batch['label'], label, indicator, din, dout, \
                           batch['item_1'].ids, batch['item_1'].segments, batch['item_1'].values, \
                           batch['item_2'].ids, batch['item_2'].segments, batch['item_2'].values, \
                           batch['item_3'].ids, batch['item_3'].segments, batch['item_3'].values, \
                           batch['item_4'].ids, batch['item_4'].segments, batch['item_4'].values, \
                           item1_grad, item2_grad, item3_grad, item4_grad, \
                           fc1_weight, fc1_bias, fc1_weight_grad, fc1_bias_grad, \
                           fc2_weight, fc2_bias, fc2_weight_grad, fc2_bias_grad])
        if result is None:
            break
        print "loss:", result[-31]
        print "prop:", result[-30]
        print "origin label:", result[-29]
        print "label:", result[-28]
        print "indicator:", result[-27]
        print "din:", result[-26] 
        print "dout:", result[-25]
        print "item_1: ids=", result[-24], "\n        segments=", result[-23], "\n        values=", result[-22]
        print "item_2: ids=", result[-21], "\n        segments=", result[-20], "\n        values=", result[-19]
        print "item_3: ids=", result[-18], "\n        segments=", result[-17], "\n        values=", result[-16]
        print "item_4: ids=", result[-15], "\n        segments=", result[-14], "\n        values=", result[-13]
        print "item1_grad", result[-12]
        print "item2_grad", result[-11]
        print "item1_grad", result[-10]
        print "item2_grad", result[-9]
        print "fc1_weight", result[-8]
        print "fc1_bias", result[-7]
        print "fc1_weight_grad", result[-6]
        print "fc1_bias_grad", result[-5]
        print "fc2_weight", result[-4]
        print "fc2_bias", result[-3]
        print "fc2_weight_grad", result[-2]
        print "fc2_bias_grad", result[-1]
        loop_num += 1

@xdl.mxnet_wrapper(is_training=True)
def model(embs, label1, comm_bs, bs):
    indicator = mx.sym.zeros(shape=(bs-comm_bs+1,))
    indicator = mx.sym.concat(indicator, mx.sym.ones(shape=(comm_bs-1,)), dim=0)
    #for i in xrange(2, comm_bs):
    #    indicator = mx.sym.concat(indicator, mx.sym.var(name='var%d' % i, shape=(1,), init=mx.init.Constant(i-1)), dim=0)
    
    ## concat, take
    din = embs[0]
    for i in xrange(1, len(embs)):
        din = mx.sym.concat(din, embs[i])
    din = mx.sym.take(din, indicator)
    
    ## fc1
    output_dim = 3
    fc1_weight = mx.sym.var(name='fc1_weight', init=mx.init.One())
    fc1_bias = mx.sym.var(name='fc1_bias', init=mx.init.Constant(0.0))
    dout = mx.sym.FullyConnected(data=din, weight=fc1_weight, bias=fc1_bias, num_hidden=output_dim, name='fc1')
    
    alpha = mx.sym.var(name='alpha_1', shape=(output_dim, ), init=mx.init.Constant(0.25))
    dout = mx.symbol.LeakyReLU(data=dout, act_type='prelu', slope=0.25, name=('prelu_1_d'), gamma=alpha)
    
    bn_gamma = mx.sym.var(name='bn_gamma_1', shape=(output_dim, ), init=mx.init.Constant(1.))
    bn_bias = mx.sym.var(name='bn_bias_1', shape=(output_dim, ), init=mx.init.Constant(0.))
    dout = mx.sym.BatchNorm(data=dout, fix_gamma=False, name=('bn_1_d'), gamma=bn_gamma, beta=bn_bias)

    ## fc2
    output_dim = 2
    fc2_weight = mx.sym.var(name='fc2_weight', init=mx.init.One())
    fc2_bias = mx.sym.var(name='fc2_bias', init=mx.init.Constant(0.0))
    dout = mx.sym.FullyConnected(data=dout, weight=fc2_weight, bias=fc2_bias, num_hidden=output_dim, name='fc2')
    
    ## label
    label0 = mx.sym.zeros(shape=(comm_bs,1))
    label1_expand = mx.sym.ones(shape=(bs-comm_bs,1))
    label0_expand = mx.sym.zeros(shape=(bs-comm_bs,1))
    label = mx.sym.concat(label0, label1)
    label_expand = mx.sym.concat(label1_expand, label0_expand)
    label = mx.sym.concat(label, label_expand, dim=0)
    
    ## prop, loss
    prop = mx.sym.SoftmaxOutput(data=dout, label=label, grad_scale=(1.0))
    loss = - mx.sym.sum(mx.sym.log(prop) * label) / bs
    return mx.sym.MakeLoss(loss), prop, label, indicator, din, dout, fc1_weight, fc1_bias, fc2_weight, fc2_bias

train()
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
