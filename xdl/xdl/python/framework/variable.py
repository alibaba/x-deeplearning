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

import contextlib

import xdl
from xdl.python.lib.graph import control_dependencies
from xdl.python.backend.model_scope import cur_model_scope
from xdl.python.utils.collections import *
from xdl.python.lib.tensor import register_converter

class VarType:
  Hash = "hash"
  Index = "index"

_VARIABLE_INFOS = [{}]

@contextlib.contextmanager
def variable_info(**kargs):
  try:
    _VARIABLE_INFOS.append(dict(kargs))
    yield
  finally:
    _VARIABLE_INFOS.pop()

def get_variable_infos():
  ret = {}
  for d in _VARIABLE_INFOS:
    ret.update(d)
  return ret

def get_variable_info(key):
  for item in _VARIABLE_INFOS[::-1]:
    if key in item:
      return item[key]
  return None

class Variable(object):
  def __init__(self, name, dtype=None, shape=None, 
               initializer=None, regularizer=None, 
               vtype=VarType.Index,
               trainable=True, collections=None, 
               scope=None,
               **kargs):
    self._name = name
    self._dtype = dtype
    self._shape = shape
    self._initializer = initializer
    self._regularizer = regularizer
    self._trainable = trainable
    self._collections = [] if collections is None else collections
    self._extra_info = dict(kargs)
    self._extra_info.update(get_variable_infos())
    self._init_value = None
    self._value = None
    self._regularizer_loss = None
    self._vtype = vtype
    self._grad_tensor = None
    self._scope = scope
    self._is_initialized_op = xdl.ps_is_initialized_op(
      var_name=self.name)
    self._do_init()

  @property
  def name(self):
    return self._name
  @property
  def dtype(self):
    return self._dtype
  @property
  def vtype(self):
    return self._vtype
  @property
  def shape(self):
    return self._shape
  @property
  def scope(self):
    return self._scope
  @property
  def initializer(self):
    return self._initializer
  @property
  def initializer_op(self):
    return self._init_value
  @property
  def is_initialized_op(self):
    return self._is_initialized_op
  @property
  def regularizer_loss(self):
    return self._regularizer_loss
  @property
  def extra_info(self):
    return self._extra_info
  @property
  def value(self):
    return self._value
  @property
  def var_register(self):
    return self._var_register
  @property
  def grad_tensor(self):
    return self._grad_tensor

  def get_extra_info(self, key, default=None):
    if key in self.extra_info:
      return self.extra_info[key]
    return default
    
  def _do_init(self):
    if self._initializer is None:
      raise Exception('initalizer not specified')

    extra_info = ''
    for k,v in self._extra_info.items():
      extra_info += '%s=%s;' % (k, v)
    self._var_register = xdl.ps_register_variable_op(
      var_name = self.name,
      var_type = self.vtype,
      shape = self.shape,
      dtype = self.dtype,
      extra_info = extra_info)

    import xdl.python.ops.initializer as initializer
    if not isinstance(self._initializer, initializer.Initializer):
      self._init_value = self._initializer
      if callable(self._initializer):
        self._init_value = self._initializer()

      import xdl.python.lib.tensor as tensor
      self._init_value = tensor.convert_to_tensor(self._init_value)

      if self._init_value is None:
        raise Exception('variable must has init value')

      if self._dtype is None:
        self._dtype = self._init_value.dtype
      if self._shape is None:
        self._shape = self._init_value.shape
      import xdl.python.ops.init_ops as init_ops
      self._init_value = init_ops.Identity(self._init_value)
    else:
      if self._dtype is None or self._shape is None:
        raise Exception("dtype and shape must be specified")
      self._init_value = self._initializer(
        self._name, self._dtype, 
        self._vtype, self._shape)

    add_to_collection(GLOBAL_INITIALIZERS, self._init_value, self._scope)
    add_to_collection(VAR_REGISTERS, self._var_register, self._scope)

    if self._trainable and TRAINABLE_VARIABLES not in self._collections:
      self._collections.append(TRAINABLE_VARIABLES)
    if GLOBAL_VARIABLES not in self._collections:
      self._collections.append(GLOBAL_VARIABLES)
    add_to_collections(self._collections, self, self._scope)

    self._value = xdl.ps_pull_op(
      var_name = self.name,
      var_type = self.vtype,
      dtype = self.dtype)

    if self._regularizer is not None:
      if not callable(self._regularizer):
        raise Exception('regularizer not callable')
      self._regularizer_loss = self._regularizer(self)
      add_to_collection(REGULARIZER_LOSS, self._regularizer_loss, self._scope)

  def gather(self, ids, save_ratio=None):
    self._grad_tensor = ids
    return xdl.ps_sparse_pull_op(
      ids,
      var_name = self.name, 
      var_type = self.vtype,
      otype = self.dtype,
      save_ratio = float(save_ratio) if save_ratio is not None else float(self.get_extra_info('save_ratio', '1.0')))

def trainable_variables():
  return trainable_variables_with_scope(['', cur_model_scope()])

def global_variables():
  return global_variables_with_scope(['', cur_model_scope()])

def global_initializers():
  return global_initializers_with_scope(['', cur_model_scope()])
    
def variable_registers():
  return variable_registers_with_scope(['', cur_model_scope()])

def trainable_variables_with_scope(scope):
  return get_collection(TRAINABLE_VARIABLES, scope)

def global_variables_with_scope(scope):
  return get_collection(GLOBAL_VARIABLES, scope)

def global_initializers_with_scope(scope):
  return get_collection(GLOBAL_INITIALIZERS, scope)
    
def variable_registers_with_scope(scope):
  return get_collection(VAR_REGISTERS, scope)    

def get_variable_by_name(name):
  for x in global_variables():
    if x.name == name:
      return x

@register_converter(Variable, 0)
def variable_to_tensor(var):
  return var.value
        
