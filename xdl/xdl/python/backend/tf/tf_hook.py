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

import os
import sys

import tensorflow as tf
from tensorflow.python.ops.variable_scope import _VariableStore

from xdl.python.framework.variable import Variable as xdl_variable
from xdl.python.backend.tf.convert_utils import TF2XDL
from xdl.python.lib.datatype import DataType
from xdl.python.utils.collections import *
from xdl.python.backend.model_scope import cur_model_scope

"""tensorflow get_variable hook"""

real_get_variable = _VariableStore.get_variable

_TF_VAR_DICT = {}

def get_variable(self, name, shape=None, dtype=DataType.float,
                 initializer=None, regularizer=None, reuse=None,
                 trainable=True, collections=None, caching_device=None,
                 partitioner=None, validate_shape=True, use_resource=None,
                 custom_getter=None, constraint=None, **kwargs):
  global _TF_VAR_DICT
  scope = cur_model_scope()
  if scope not in _TF_VAR_DICT:
    _TF_VAR_DICT[scope] = {}
  tf_var_dict = _TF_VAR_DICT[scope]
  if name in tf_var_dict:
    if tf.get_variable_scope().reuse in [True, tf.AUTO_REUSE]:
      return tf_var_dict[name]
    else:
      raise Exception("must set reuse flag to enable reuse")

  def _custom_getter(getter, *args, **kwargs):
    tf_var = getter(*args, **kwargs)
    xdl_var = xdl_variable(
      name = name,
      shape = TF2XDL.convert_shape(shape),
      dtype = TF2XDL.convert_type(dtype),
      scope = scope,
      trainable = True,
      initializer = TF2XDL.convert_initializer(initializer))
    add_to_collection(VAR_MAPPING, (xdl_var, tf_var), scope)
    add_to_collection(BACKPROP_VARS, (name, tf_var), scope)
    tf_var_dict[name] = tf_var
    return tf_var

  return real_get_variable(self, name, shape, dtype, initializer, 
                           regularizer, reuse, trainable,
                           collections, caching_device, partitioner, 
                           validate_shape, use_resource, _custom_getter, 
                           constraint, **kwargs)

_VariableStore.get_variable = get_variable    
