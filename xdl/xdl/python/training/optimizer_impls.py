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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xdl
from xdl.python.training.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate):
        """construct a SGD optimizer
           Args:
             learning_rate: a float value indicate learning rate
        """
        super(SGD, self).__init__()
        self._lr = learning_rate

    def dense_update(self, var, grad):
        return xdl.ps_dense_apply_momentum_op(
            learning_rate = self._lr,
            momentum = 0.0,
            grad = grad,
            var_name = var.name,
            var_type = var.vtype,
            use_nesterov = False)

    def sparse_update(self, var, grad, indices):
        return xdl.ps_sparse_apply_momentum_op(
            learning_rate = self._lr,
            momentum = 0.0,
            grad = grad,
            indices = indices,
            var_name = var.name,
            var_type = var.vtype,
            use_nesterov = False)

class Momentum(Optimizer):
    def __init__(self, learning_rate, momentum, 
                  use_nesterov=False):
        """construct a Momentum optimizer
           Args:
             learning_rate: a float value indicate learning rate
             momentum: a float value indicate momentum
             use_nesterov: a bool value, if true use nesterov momentum
        """
        super(Momentum, self).__init__()
        self._lr = learning_rate
        self._momentum = momentum
        self._use_nesterov = use_nesterov

    def dense_update(self, var, grad):
        return xdl.ps_dense_apply_momentum_op(
            learning_rate = self._lr,
            momentum = self._momentum,
            grad = grad,
            var_name = var.name,
            var_type = var.vtype,
            use_nesterov = self._use_nesterov)

    def sparse_update(self, var, grad, indices):
        return xdl.ps_sparse_apply_momentum_op(
            learning_rate = self._lr,
            momentum = self._momentum,
            grad = grad,
            indices = indices,
            var_name = var.name,
            var_type = var.vtype,
            use_nesterov = self._use_nesterov)

class Adagrad(Optimizer):
    def __init__(self, learning_rate, initial_accumulator_value=0.1):
        """construct a adagrad optimizer
           Args:
             learning_rate: a float value indicate learning rate
             initial_accumulator_value: a float value, start value of accumulators, must be positive
        """
        super(Adagrad, self).__init__()
        self._lr = learning_rate
        self._init_acc = initial_accumulator_value
    
    def dense_update(self, var, grad):
        return xdl.ps_dense_apply_adagrad_op(
            learning_rate = self._lr,
            grad = grad,
            initial_accumulator_value = self._init_acc,
            var_name = var.name,
            var_type = var.vtype)

    def sparse_update(self, var, grad, indices):
        return xdl.ps_sparse_apply_adagrad_op(
            learning_rate = self._lr,
            initial_accumulator_value = self._init_acc,
            grad = grad,
            indices = indices,
            var_name = var.name,
            var_type = var.vtype)

class Adam(Optimizer):
    def __init__(self, 
                 learning_rate=0.001, 
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 lr_decay=True):
        """construct a adam optimizer
           Args:
             learning_rate: a float value indicate learning rate
             beta1: a float value, the exponential decay rate for the 1st moment estimates
             beta2: a float value, the exponential decay rate for the 2nd moment estimates
             epsilon: a small constant for numerical stability
        """
        super(Adam, self).__init__()
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._lr_decay = lr_decay
    
    def dense_update(self, var, grad):
        return xdl.ps_dense_apply_adam_op(
            beta1 = self._beta1,
            beta2 = self._beta2,
            epsilon = self._epsilon,
            learning_rate = self._lr,
            lr_decay = self._lr_decay,
            grad = grad,
            var_name = var.name,
            var_type = var.vtype)

    def sparse_update(self, var, grad, indices):
        return xdl.ps_sparse_apply_adam_op(
            beta1 = self._beta1,
            beta2 = self._beta2,
            epsilon = self._epsilon,
            learning_rate = self._lr,
            lr_decay = self._lr_decay,
            grad = grad,
            indices = indices,
            var_name = var.name,
            var_type = var.vtype)

class Ftrl(Optimizer):
    def __init__(self, 
                 learning_rate, 
                 learning_rate_power=-0.5,
                 initial_accumulator_value=0.1,
                 l1_regularization_strength=0.0,
                 l2_regularization_strength=0.0):
        """construct a ftrl optimizer
           Args:
             learning_rate: a float value indicate learning rate
             learning_rate_power: a float value, must be less or equal to zero
             initial_accumulator_value: the starting value for accumulators
             l1_regularization_strength: a float value, must be greater than or equal to zero
             l2_regularization_strength: a float value, must be greater than or equal to zero
        """
        super(Ftrl, self).__init__()
        self._lr = learning_rate
        self._lr_power = learning_rate_power
        self._init_acc = initial_accumulator_value
        self._l1_reg = l1_regularization_strength
        self._l2_reg = l2_regularization_strength
    
    def dense_update(self, var, grad):
        return xdl.ps_dense_apply_ftrl_op(
            learning_rate = self._lr,
            learning_rate_power = self._lr_power,
            initial_accumulator_value = self._init_acc,
            l1_reg = self._l1_reg,
            l2_reg = self._l2_reg,
            grad = grad,
            var_name = var.name,
            var_type = var.vtype)

    def sparse_update(self, var, grad, indices):
        return xdl.ps_sparse_apply_ftrl_op(
            learning_rate = self._lr,
            learning_rate_power = self._lr_power,
            initial_accumulator_value = self._init_acc,
            l1_reg = self._l1_reg,
            l2_reg = self._l2_reg,
            grad = grad,
            indices = indices,
            var_name = var.name,
            var_type = var.vtype)
