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

import xdl
from xdl.python.ops.initializer import Initializer

_DEFAULT_SEED = 100

class IdentityInitializer(Initializer):
    def __init__(self, value):
        super(IdentityInitializer, self).__init__()
        self.value_ = value

    def __call__(self, var_name, dtype, vtype, shape):
        return xdl.ps_identity_initializer_op(
            var_name = var_name,
            var_type = vtype,
            value = self.value_)

class ZeroInitializer(Initializer):
    def __init__(self):
        super(ZeroInitializer, self).__init__()

    def __call__(self, var_name, dtype, vtype, shape):
        return xdl.ps_constant_initializer_op(
            var_name = var_name,
            var_type = vtype,
            value = 0.0)

class OneInitializer(Initializer):
    def __init__(self):
        super(OneInitializer, self).__init__()

    def __call__(self, var_name, dtype, vtype, shape):
        return xdl.ps_constant_initializer_op(
            var_name = var_name,
            var_type = vtype,
            value = 1.0)

class ConstantInitializer(Initializer):
    def __init__(self, value):
        super(ConstantInitializer, self).__init__()
        self._value = value

    def __call__(self, var_name, dtype, vtype, shape):
        return xdl.ps_constant_initializer_op(
            var_name = var_name,
            var_type = vtype,
            value = self._value)

class RandomNormalInitializer(Initializer):
    def __init__(self, mean=0.0, stddev=1.0, seed=None):
        super(RandomNormalInitializer, self).__init__()
        self._mean = mean
        self._stddev = stddev
        self._seed = _DEFAULT_SEED if seed is None else seed

    def __call__(self, var_name, dtype, vtype, shape):
      return xdl.ps_normal_initializer_op(
          var_name=var_name,
          var_type=vtype,
          seed=self._seed,
          mean=self._mean,
          stddev=self._stddev
      )

class RandomUniformInitializer(Initializer):
    def __init__(self, minval=0.0, maxval=None, seed=None):
        super(RandomUniformInitializer, self).__init__()
        self._minval = minval
        self._maxval = maxval
        self._seed = _DEFAULT_SEED if seed is None else seed

    def __call__(self, var_name, dtype, vtype, shape):
        raise Exception('unimplement')

class TruncatedNormalInitializer(Initializer):
    def __init__(self, mean=0.0, stddev=1.0, seed=None):
        super(TruncatedNormalInitializer, self).__init__()
        self._mean = mean
        self._stddev = stddev
        self._seed = _DEFAULT_SEED if seed is None else seed

    def __call__(self, var_name, dtype, vtype, shape):
        return xdl.ps_truncated_normal_initializer_op(
            var_name = var_name,
            var_type = vtype,
            seed = self._seed,
            mean = self._mean,
            stddev = self._stddev)

class NormalInitializer(Initializer):
    def __init__(self, mean=0.0, stddev=1.0, seed=None):
        super(NormalInitializer, self).__init__()
        self._mean = mean
        self._stddev = stddev
        self._seed = _DEFAULT_SEED if seed is None else seed

    def __call__(self, var_name, dtype, vtype, shape):
        return xdl.ps_normal_initializer_op(
            var_name = var_name,
            var_type = vtype,
            seed = self._seed,
            mean = self._mean,
            stddev = self._stddev)

class UniformUnitScalingInitializer(Initializer):
    def __init__(self, factor=1.0, seed=None):
        super(UniformUnitScalingInitializer, self).__init__()
        self._factor = factor
        self._seed = _DEFAULT_SEED if seed is None else seed

    def __call__(self, var_name, dtype, vtype, shape):
        return xdl.ps_uniform_unit_scaling_initializer_op(
            var_name = var_name,
            var_type = vtype,
            seed = self._seed,
            factor = self._factor,
            shape = shape)

class VarianceScalingInitializer(Initializer):
    def __init__(self, scale=1.0,
                 mode='fan_in',
                 distribution='truncated_normal',
                 seed=None):
        super(VarianceScalingInitializer, self).__init__()
        self._scale = scale
        self._mode = mode
        self._distribution = distribution
        self._seed = _DEFAULT_SEED if seed is None else seed

    def __call__(self, var_name, dtype, vtype, shape):
        return xdl.ps_variance_scaling_initializer_op(
            var_name = var_name,
            var_type = vtype,
            seed = self._seed,
            scale = self._scale,
            mode = self._mode,
            distribution = self._distribution,
            shape = shape)

class OrthogonalInitializer(Initializer):
    def __init__(self, shape, gain=1.0, seed=None):
        super(OrthogonalInitializer, self).__init__()
        if len(shape) < 2:
            raise Exception('dimension must > 1')
        self._dim = 1
        for dim in shape[1:]:
            self._dim *= dim
        self._gain = gain
        self._seed = _DEFAULT_SEED if seed is None else seed

    def __call__(self, var_name, dtype, vtype, shape):
        return xdl.ps_orthogonal_initializer_op(
            var_name = var_name,
            var_type = vtype,
            seed = self._seed,
            gain = self._gain,
            dim = self._dim)





