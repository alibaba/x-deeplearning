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

import mxnet as mx

def copy_data(executor):
    embs = []
    embs.append(mx.nd.array([[1.], [1.], [1.], [1.]]))
    embs.append(mx.nd.array([[1.], [1.], [1.], [1.]]))
    embs.append(mx.nd.array([[1.], [1.], [1.], [1.]]))
    embs.append(mx.nd.array([[2.], [2.], [2.], [2.]]))
    for i in xrange(len(embs)):
        embs[i].copyto(executor.arg_dict['emb%d' % (i+1)])

    label = mx.nd.array([[0., 1.], [0., 1.], [0., 1.], [0., 1.], [1., 0.], [1., 0.], [1., 0.]])
    label.copyto(executor.arg_dict['label'])

    fc1_weight = mx.nd.array([[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]])
    fc1_weight.copyto(executor.arg_dict['fc1_weight'])
    fc1_bias = mx.nd.array([0., 0., 0.])
    fc1_bias.copyto(executor.arg_dict['fc1_bias'])

    fc2_weight = mx.nd.array([[1., 1., 1.], [1., 1., 1.]])
    fc2_weight.copyto(executor.arg_dict['fc2_weight'])
    fc2_bias = mx.nd.array([0., 0.])
    fc2_bias.copyto(executor.arg_dict['fc2_bias'])

    print "executor.arg_dict:", executor.arg_dict
    print "executor.grad_dict:", executor.grad_dict
    print "executor.aux_dict:", executor.aux_dict

def train():
    model_outputs = model(embs_len=4, comm_bs=4, bs=7)
    symbol_list = []
    symbol_list.append(model_outputs[0])
    symbol_list.append(model_outputs[1])
    if len(model_outputs) > 2:
        symbol_list.extend([mx.sym.BlockGrad(x) for x in model_outputs[2:]])
    symbol = mx.sym.Group(symbol_list)
    print 'symbol.tojson() =', symbol.tojson()
    
    executor = symbol.simple_bind(ctx=mx.cpu(), grad_req='write')
    
    for itr in xrange(1, 2):
        print "<<<<<<<", itr, ">>>>>>>>"
        copy_data(executor)
        executor.forward(is_train=True)
        print "loss:", executor.outputs[0]
        print "prop:", executor.outputs[1]
        print "label:", executor.outputs[2]
        print "indicator:", executor.outputs[3]
        print "din:", executor.outputs[4]
        print "dout:", executor.outputs[5]
        print "fc1_weight:", executor.outputs[6]
        print "fc1_bias:", executor.outputs[7]
        print "fc2_weight:", executor.outputs[8]
        print "fc2_bias:", executor.outputs[9]
        executor.backward()
        print "\n## args:", executor.arg_dict
        print "\n## grads:", executor.grad_dict
        print "\n## auxs:", executor.aux_dict
    
def model(embs_len, comm_bs, bs):
    embs = []
    for i in xrange(embs_len):
        embs.append(mx.sym.var(name='emb%d' % (i+1), shape=[comm_bs, 1]))
    label = mx.sym.var(name='label', shape=[bs, 2])
    label = mx.sym.BlockGrad(label)
    
    indicator = mx.sym.zeros(shape=(bs-comm_bs+1,))
    indicator = mx.sym.concat(indicator, mx.sym.ones(shape=(comm_bs-1,)), dim=0)
    
    ## concat, take
    din = embs[0]
    for i in xrange(1, embs_len):
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

    ## prop, loss
    prop = mx.sym.SoftmaxOutput(data=dout, label=label, grad_scale=(1.0))
    loss = - mx.sym.sum(mx.sym.log(prop) * label) / bs
    return mx.sym.MakeLoss(loss), prop, label, indicator, din, dout, fc1_weight, fc1_bias, fc2_weight, fc2_bias

train()
