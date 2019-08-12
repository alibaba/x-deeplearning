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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import mxnet as mx
import numpy as np

from mxnet.initializer import One
from mxnet.initializer import Zero
from mxnet.initializer import Constant

from xdl.python.lib.datatype import DataType as xt
from xdl.python.lib.tensorshape import TensorShape as xts

class MX2XDL(object):
    @staticmethod
    def convert_shape(shape):
        return xts(list(shape))

    @staticmethod
    def convert_type(dtype):
        if dtype == np.int16:
            return xt.int16
        if dtype == np.int32:
            return xt.int32
        elif dtype == np.int64:
            return xt.int64
        elif dtype == np.float32 or dtype is None:
            return xt.float
        elif dtype == np.float64:
            return xt.double
        else:
            raise Exception("unsupported datatype:", dtype)

    @staticmethod
    def convert_initializer(initializer, args):
        import xdl.python.ops.init_ops as xi
        if initializer is None or initializer == '':
            return xi.Zeros()
        elif initializer == 'one':
            return xi.Ones()
        elif initializer == 'zero':
            return xi.Zeros()
        elif initializer == 'constant' or initializer == 'Constant':
            return xi.Constant(value=args['value'])
        elif initializer == 'uniform':
            scale = 0.07
            if args.has_key('scale'):
                scale = args['scale']
            return xi.UniformUnitScaling(factor=scale)
        elif initializer == 'normal':
            sigma = 0.01
            if args.has_key('sigma'):
                sigma = args['sigma']
            return xi.TruncatedNormal(stddev=sigma)
        elif initializer == 'identity':
            param = []
            if args.has_key('init_value'):
                param = args['init_value']
            return xi.Identity(np.array(param, dtype=np.float32))
        else:
            raise Exception('unsupport mxnet initializer:' + initializer)


class XDL2MX(object):
    @staticmethod
    def convert_type(dtype):
        if dtype == xt.int16:
            return 'int16'
        elif dtype == xt.int32:
            return 'int32'
        elif dtype == xt.int64:
            return 'int64'
        elif dtype == xt.float:
            return 'float32'
        elif dtype == xt.double:
            return 'float64'
        else:
            raise Exception("unsupported datatype:", dtype)
