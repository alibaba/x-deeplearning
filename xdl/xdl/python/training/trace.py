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

from xdl.python.utils.collections import *
from xdl.python.lib.tensor import Tensor
from xdl.python.framework.variable import Variable
from xdl.python.training.gradient_utils import get_gradient

TRACE_INFOS = "trace_infos"

class TraceInfo(object):
  def __init__(self, info):
    self._info = info

  @property
  def name(self):
    assert 'name' in self._info
    return self._info['name']

  @property
  def tensor(self):
    assert 'tensor' in self._info
    return self._info['tensor']

  @property
  def value(self):
    assert 'value' in self._info
    return self._info['value']

  @property
  def vtype(self):
    assert 'vtype' in self._info
    return self._info['vtype']

  @property
  def function(self):
    assert 'function' in self._info
    return self._info['function']

  @property
  def variable(self):
    assert 'variable' in self._info
    return self._info['variable']

  def set_value(self, value):
    self._info['value'] = value

def get_names(vtypes, scope=None):
  info = get_collection(TRACE_INFOS, scope)
  if info is None:
    return []
  vtypes = list(vtypes) if isinstance(vtypes, (list, tuple)) else [vtypes]
  return [v.name for v in info if v.vtype in vtypes]

def get_values(vtypes, scope=None):
  info = get_collection(TRACE_INFOS, scope)
  if info is None:
    return []
  vtypes = list(vtypes) if isinstance(vtypes, (list, tuple)) else [vtypes]
  return [v.value for v in info if v.vtype in vtypes]

def get_variables(vtypes, scope=None):
  info = get_collection(TRACE_INFOS, scope)
  if info is None:
    return []
  vtypes = list(vtypes) if isinstance(vtypes, (list, tuple)) else [vtypes]
  return [v.variable for v in info if v.vtype in vtypes]

def get_tensors(vtypes, scope=None):
  info = get_collection(TRACE_INFOS, scope)
  if info is None:
    return []
  vtypes = list(vtypes) if isinstance(vtypes, (list, tuple)) else [vtypes]
  return [v.tensor for v in info if v.vtype in vtypes]

def get_functions(vtypes, scope=None):
  info = get_collection(TRACE_INFOS, scope)
  if info is None:
    return []
  vtypes = list(vtypes) if isinstance(vtypes, (list, tuple)) else [vtypes]
  return [v.function for v in info if v.vtype in vtypes]

def set_values(vtypes, values, scope=None):
  info = get_collection(TRACE_INFOS, scope)
  if info is None:
    assert len(values) == 0
    return
  vtypes = list(vtypes) if isinstance(vtypes, (list, tuple)) else [vtypes]
  k = 0
  for v in info:
    if v.vtype in vtypes:
      v.set_value(values[k])
      k += 1
  assert k == len(values)

def trace_tensor(name, t, scope=None, backend='xdl', summary=None, variable=None):
  info = {'name': name}
  if backend in ['tf', 'tf_sparse_assign']:
    import tensorflow as tf
    assert isinstance(t, (tf.Tensor, tf.Variable))
    t = tf.identity(t)
  elif backend == 'mxnet':
    import mxnet as mx
    assert isinstance(t, mx.sym.Symbol)
    assert name == t.name, 'mxnet symbol names must keep in trace'
    t = mx.sym.BlockGrad(t)
  else:
    assert isinstance(t, Tensor)
    info['value'] = t
  info['tensor'] = t
  info['vtype'] = backend
  info['function'] = summary
  info['variable'] = variable
  add_to_collection(TRACE_INFOS, TraceInfo(info), scope)

def trace_tf_tensor(name, value, scope=None, summary=None):
  trace_tensor(name, value, scope, backend='tf', summary=summary)

def trace_mxnet_tensor(name, value, scope=None, summary=None):
  trace_tensor(name, value, scope, backend='mxnet', summary=summary)

def trace_variable(name, var, scope=None, summary=None):
  trace_tensor(name, var.value, scope=scope, backend='xdl', summary=summary)

def trace_gradient(name, key=None, scope=None, summary=None):
  if key is None:
    if scope is not None and scope != '':
      key = scope + '/gradient/' + name
    else:
      key = 'gradient/' + name
  gradient = get_gradient(name, scope)
  trace_tensor(key, gradient, scope=scope, backend='xdl', summary=summary)

def trace_collection(collection, scope=None):
  coll = get_collection(collection, scope)
  for var in coll:
    key = var.name
    if scope is not None and scope != '':
      key = scope + '/' + key
    trace_tensor(key, var.value, scope=scope, backend='xdl')

def trace_callback(name, callback, scope=None):
  info = {'name': name, 'vtype': 'function', 'function': callback}
  add_to_collection(TRACE_INFOS, TraceInfo(info), scope)

def trace_once(name, value, scope=None, summary=None):
  assert isinstance(value, Tensor)
  info = {'name': name, 'vtype': 'once', 'tensor': value, 'value': value}
  add_to_collection(TRACE_INFOS, TraceInfo(info), scope)

def trace_sparse_assign(name, ids, values, shape, dtype, vtype, scope=None):
  import tensorflow as tf
  from xdl.python.ops.init_ops import Zeros
  assert isinstance(ids, tf.Tensor)
  assert isinstance(values, tf.Tensor)
  shape = [1] + list(shape)
  var = Variable(name=name,shape=shape,dtype=dtype,initializer=Zeros(),
      trainable=False,vtype=vtype,scope=scope)
  trace_tensor(name + '_ids', ids, scope=scope, backend='tf_sparse_assign',
      variable=var)
  trace_tensor(name + '_values', values, scope=scope,
      backend='tf_sparse_assign', variable=var)
