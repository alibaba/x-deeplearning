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

import numpy as np
import mxnet as mx
import math


class Identity(mx.init.Initializer):
    def __init__(self, init_value=None):
        super(Identity, self).__init__(init_value=init_value)


class mx_dnn_layer(object):
    def __init__(self, input_dim, output_dim, active_op='prelu', use_batch_norm=False, version="default"):

        self.active_op = active_op
        self.use_batch_norm = use_batch_norm
        self.version = version 
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_out = None
        self.out = None

        self.w = None
        self.b = None
        self.alpha = None

        # BatchNorm Param
        self.bn_gamma = None
        self.bn_bias = None

        # Moving mean/var
        self.bn_moving_mean = None
        self.bn_moving_var = None

    def call(self, bottom_data):

        init_mean = 0.0
        init_stddev = 1.   # 0.001
        init_value = (init_stddev * np.random.randn(self.output_dim, self.input_dim).astype(np.float32) + init_mean) / np.sqrt(self.input_dim)
        self.w = mx.sym.var(name='fc_w_%s' % self.version, init=Identity(init_value=init_value.tolist()))
        self.b = mx.sym.var(name='fc_b_%s' % self.version, init=mx.init.Constant(0.1))
        self.out = mx.symbol.FullyConnected(data=bottom_data, name=('fc_%s' % self.version), num_hidden=self.output_dim, weight=self.w, bias=self.b)
        print "if mx.symbol.FullyConnected"

        if self.use_batch_norm:
            print "if self.use_batch_norm:"
            # BatchNorm Param
            self.bn_gamma = mx.sym.var(name='bn_gamma_1_%s' % self.version, shape=(self.output_dim, ), init=mx.init.Constant(1.))
            self.bn_bias = mx.sym.var(name='bn_bias_1_%s' % self.version, shape=(self.output_dim, ), init=mx.init.Constant(0.))

            # Moving mean/var
            self.bn_moving_mean = mx.sym.zeros((self.output_dim,) )
            self.bn_moving_var = mx.sym.zeros((self.output_dim,) )
            self.out = mx.symbol.BatchNorm(data=self.out, fix_gamma=False, name=('bn_%s' % self.version), gamma=self.bn_gamma, beta=self.bn_bias)

        if self.active_op == 'prelu':
            print "if self.active_op == 'prelu':"
            self.alpha = mx.sym.var(name='alpha_1_%s' % self.version, shape=(self.output_dim, ), init=mx.init.Constant(0.25))
            self.out = mx.symbol.LeakyReLU(data=self.out, act_type='prelu', slope=-0.25, name=('prelu_%s' % self.version), gamma=self.alpha)

        return self.out

class FullyConnected3D(object):
    def __init__(self, input_dim, output_dim, active_op='prelu', version="default", batch_size=5000):
        
        self.active_op = active_op
        self.version = version 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.weight = None
        self.bias = None
        self.alpha = None
        
    def call(self, bottom_data):

        print "call FullyConnected3D"
        init_mean = 0.0
        init_stddev = 1. 
        init_value = (init_stddev * np.random.randn(1, self.input_dim, self.output_dim).astype(np.float32) + init_mean) / np.sqrt(self.input_dim)
        self.weight = mx.sym.var(name='fc_w_%s' % self.version, init=Identity(init_value=init_value.tolist()), shape=(1, self.input_dim, self.output_dim))

        b_value = 0.1
        self.bias = mx.sym.var(name='fc_b_%s' % self.version, shape=(1, self.output_dim), init=mx.init.Constant(0.1))

        self.weight = mx.symbol.broadcast_to(data=self.weight, shape=(self.batch_size, 0, 0))
        net_dot = mx.symbol.batch_dot(lhs=bottom_data, rhs=self.weight)
        net_dot = mx.symbol.broadcast_plus(lhs=net_dot, rhs=self.bias)

        if self.active_op == 'prelu':
            print "in FullyConnected3D use prelu"
            self.alpha = mx.sym.var(name='alpha_1_%s' % self.version, shape=(1, 1, self.output_dim), init=mx.init.Constant(0.25)) 
            r1 = mx.symbol.Activation(data=net_dot, act_type='relu')
            r2 = mx.symbol.Activation(data=-net_dot, act_type='relu')
            net_dot = mx.symbol.ElementWiseSum(*[r1, - mx.symbol.broadcast_mul(lhs=r2, rhs=self.alpha)])
        
        return net_dot

