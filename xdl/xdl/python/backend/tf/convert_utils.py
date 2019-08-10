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

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops import Constant
from tensorflow.python.ops.init_ops import Ones
from tensorflow.python.ops.init_ops import Zeros
from tensorflow.python.ops.init_ops import RandomNormal
from tensorflow.python.ops.init_ops import RandomUniform
from tensorflow.python.ops.init_ops import TruncatedNormal
from tensorflow.python.ops.init_ops import UniformUnitScaling
from tensorflow.python.ops.init_ops import VarianceScaling

from xdl.python.lib.datatype import DataType as xt
from xdl.python.lib.tensorshape import TensorShape as xts

class XDL2TF(object):
    @staticmethod
    def convert_shape(shape):
        return tf.TensorShape(shape.dims())

    @staticmethod
    def convert_type(dtype):
        if dtype == xt.int16:
            return tf.int16
        if dtype == xt.int32:
            return tf.int32
        elif dtype == xt.int64:
            return tf.int64
        elif dtype == xt.float:
            return tf.float32
        elif dtype == xt.double:
            return tf.float64
        else:
            raise Exception("unknown xdl dtype:", dtype)

class TF2XDL(object):
    @staticmethod
    def convert_shape(shape):
        if isinstance(shape, tf.TensorShape):
            return xts(shape.as_list())
        else:
            return xts(shape)

    @staticmethod
    def convert_type(dtype):
        if dtype == tf.int16:
            return xt.int16
        if dtype == tf.int32:
            return xt.int32
        elif dtype == tf.int64:
            return xt.int64
        elif dtype == tf.float32:
            return xt.float
        elif dtype == tf.float64:
            return xt.double
        else:
            raise Exception("unsupport tf dtype:", dtype)

    @staticmethod
    def convert_initializer(initializer):
        import xdl.python.ops.init_ops as xi
        if initializer == None:
            return xi.Null
        elif type(initializer) == Constant:
            value = getattr(initializer, 'value')
            if isinstance(value, (np.ndarray, np.generic)):
                return xi.Identity(value=value)
            return xi.Constant(value)
        elif type(initializer) == Ones:
            return xi.Ones()
        elif type(initializer) == Zeros:
            return xi.Zeros()
        elif type(initializer) == RandomNormal:
            mean = getattr(initializer, 'mean')
            stddev = getattr(initializer, 'stddev')
            seed = getattr(initializer, 'seed')
            return xi.RandomNormal(mean, stddev, seed)
        elif type(initializer) == RandomUniform:
            minval = getattr(initializer, 'minval')
            maxval = getattr(initializer, 'maxval')
            seed = getattr(initializer, 'seed')
            return xi.RandomUniform(minval, maxval, seed)
        elif type(initializer) == TruncatedNormal:
            mean = getattr(initializer, 'mean')
            stddev = getattr(initializer, 'stddev')
            seed = getattr(initializer, 'seed')
            return xi.TruncatedNormal(mean, stddev, seed)
        elif type(initializer) == UniformUnitScaling:
            factor = getattr(initializer, 'factor')
            seed = getattr(initializer, 'seed')
            return xi.UniformUnitScaling(factor, seed)
        elif type(initializer) == VarianceScaling:
            scale = getattr(initializer, 'scale')
            mode = getattr(initializer, 'mode')
            distribution = getattr(initializer, 'distribution')
            seed = getattr(initializer, 'seed')
            return xi.VarianceScaling(scale, mode, distribution, seed)
        else:
            raise Exception('unsupport tf initializer:' + str(initializer))

