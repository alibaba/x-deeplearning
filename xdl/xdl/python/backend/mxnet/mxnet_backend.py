# Copyright 2018 Alibaba Group. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 (the "License");
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
import json
import xdl

import mxnet as mx
import numpy as np

import xdl.python.sparse_engine.embedding
import xdl.python.framework.variable
import xdl.python.backend.mxnet.batchnorm_hook
from xdl.python.training.gradient_utils import set_gradients, set_gear_gradient, add_var_mapping
from xdl.python.lib.tensor import Tensor
from xdl.python.utils.collections import *
from xdl.python.backend.model_scope import cur_model_scope
from xdl.python.training import trace
import random
from xdl.python.backend.backend_type import set_backend_type

"""python adapter for mxnet."""

_INPUT_NAME_DICT = set([])
_GEAR_INPUTS = []
def create_name(prefix):
  global _INPUT_NAME_DICT
  if prefix not in _INPUT_NAME_DICT:
    _INPUT_NAME_DICT.add(prefix)
    return prefix
  i = 0
  name = prefix + "_" + str(i)
  while name in _INPUT_NAME_DICT:
    i += 1
    name = prefix + "_" + str(i)
  _INPUT_NAME_DICT.add(name)
  return name

def recursive_make_placeholder(x, sym_input_dict, gear=False):
  if isinstance(x, (tuple, list)):
    return [recursive_make_placeholder(item, sym_input_dict, gear) for item in x]
  elif isinstance(x, Tensor):
    return make_placeholder(x, sym_input_dict, gear)
  else:
    return x

def make_placeholder(x, sym_input_dict, gear):
  """define a tensorflow placeholder for xdl input x.
  Args:
  x: xdl embedding info returned by xdl.embedding()
  Returns:
  a tf placeholder
  Raises:
  None
  """
  import xdl.python.sparse_engine.embedding as emb
  if x.shape is None or len(x.shape) == 0:
    raise Exception("no shape info")
  emb_info = emb.get_embedding_info(x)
  if emb_info is not None:
    emb_info._output_tensor = x
    name = create_name(emb_info.name)
    add_var_mapping(emb_info.var, name)
    sym_input_dict[name] = x
    return mx.sym.Variable(
      name,
      shape=[x.shape[0], emb_info.emb_dim],
      dtype='float32',
      __create_by_xdl__=True)
  else:
    global _GEAR_INPUTS
    name = create_name("input")
    if gear:
      name += '_gear'
      _GEAR_INPUTS.append(name)
    import xdl.python.backend.mxnet.convert_utils as cu
    sym_input_dict[name] = x
    return mx.sym.Variable(
      name,
      shape=x.shape,
      dtype=cu.XDL2MX.convert_type(x.dtype),
      __create_by_xdl__=True)

def serialize_graph(symbol):
  """serialize mxnet graph to json."""
  return symbol.tojson()

def add_variable_inputs(sym, sym_input_dict, is_training):
  arg_shape, _, aux_shape = sym.infer_shape()
  arg_type, _, aux_type = sym.infer_type()
  arg_shape_type = zip(sym.list_arguments(),
                       arg_shape,
                       arg_type)
  aux_shape_type = zip(sym.list_auxiliary_states(),
                       aux_shape,
                       aux_type)
  node_info = json.loads(sym.tojson())["nodes"]
  for item in arg_shape_type:
    import xdl.python.framework.variable as variable
    import xdl.python.backend.mxnet.convert_utils as cu
    if item[0] not in sym_input_dict:
      initializer_and_args = get_initializer_and_args(
        item[0], node_info)
      xdl_var = variable.Variable(
        name = item[0],
        shape = item[1],
        dtype = cu.MX2XDL.convert_type(item[2]),
        trainable = True,
        initializer = cu.MX2XDL.convert_initializer(
          initializer_and_args[0],
          initializer_and_args[1]))
      sym_input_dict[item[0]] = xdl_var.value

  for item in aux_shape_type:
    if item[0].endswith('_moving_mean') or \
          item[0].endswith('_moving_var'):
      initializer_and_args = get_initializer_and_args(
        item[0], node_info)
      xdl_var = variable.Variable(
        name = item[0],
        shape = item[1],
        dtype = cu.MX2XDL.convert_type(item[2]),
        trainable = True,
        initializer = cu.MX2XDL.convert_initializer(
          initializer_and_args[0],
          initializer_and_args[1]))
      if not is_training:
        sym_input_dict[item[0]] = xdl_var.value

def get_initializer_and_args(name, node_infos):
  for node in node_infos:
    if name == node['name']:
      if 'attrs' not in node or '__init__' not in node['attrs']:
        return [None, None]
      else:
        init_info = json.loads(node['attrs']['__init__'])
        return init_info
  return [None, None]

def flatten(input):
  if isinstance(input, list):
    return flatten_list(input)
  else:
    return [input]

def flatten_list(inputs):
  output = []
  for x in inputs:
    if isinstance(x, list):
      output.extend(flatten_list(x))
    else:
      output.append(x)
  return output

def get_symbol_list(model_outputs):
  symbol_list = list(model_outputs)
  bn_statistic = get_collection(MXNET_BN_STATISTIC)
  bn_var_names = []
  bn_syms = []
  moments = []
  if bn_statistic is not None and len(bn_statistic) > 0:
    bn_var_names.extend([x[0] for x in bn_statistic])
    bn_syms.extend([x[1] for x in bn_statistic])
    moments.extend([x[2] for x in bn_statistic])

  symbol_list.extend([mx.sym.BlockGrad(x) for x in bn_syms])
  return symbol_list, bn_var_names, bn_syms, moments

def get_trace_outputs(sym):
  args = sym.list_arguments()
  auxs = sym.list_auxiliary_states()
  trace_names = trace.get_names('mxnet')
  trace_syms = trace.get_tensors('mxnet')
  res = []
  for i in xrange(len(trace_names)):
    name = trace_names[i]
    sym_name = trace_syms[i].name
    if name in args or name in auxs or sym_name in args or sym_name in auxs:
      continue
    res.append(trace_syms[i])
  return res

def set_trace_outputs(sym_input_dict, outputs):
  trace_output = []
  trace_names = trace.get_names('mxnet')
  trace_syms = trace.get_tensors('mxnet')
  k = 0
  for i in xrange(len(trace_names)):
    name = trace_names[i]
    sym_name = trace_syms[i].name
    if name in sym_input_dict:
      trace_output.append(sym_input_dict[name])
    elif sym_name in sym_input_dict:
      trace_output.append(sym_input_dict[sym_name])
    else:
      trace_output.append(outputs[k])
      k += 1
  assert k == len(outputs)
  trace.set_values('mxnet', trace_output)

def mxnet_wrapper(device_type='cpu', is_training=True, init_grad=None):
  """python decorator to adapt a mxnet-model define function to xdl.

  Args:
  device_type: on which device the mxnet-model whill run, can only be cpu/gpu

  Returns:
  a decorator

  Raises:
  raise exception when model_func return none
  """
  def decorator(model_func):
    """ model_func: a function define a mxnet model using native mxnet api
    return value must be loss
    """
    def _wrapper(*inputs, **kwargs):
      set_backend_type('mxnet')
      add_to_collection(BACKEND_DEVICE_TYPE, device_type.lower())
      sym_input_dict = {}
      placeholders = []
      for x in inputs:
        placeholder = recursive_make_placeholder(x, sym_input_dict)
        placeholders.append(placeholder)

      gear_input_num = 0
      if 'gear_inputs' in kwargs:
        gear_inputs = kwargs['gear_inputs']
        gear_placeholder = recursive_make_placeholder(gear_inputs, sym_input_dict, True)
        kwargs['gear_inputs'] = gear_placeholder
        gear_input_num = len(flatten(gear_inputs))

      model_outputs = model_func(*placeholders, **kwargs)
      if len(model_outputs) == 0:
        raise Exception('model_func must return loss')
      symbol_list, bn_var_names, bn_syms, moments = get_symbol_list(model_outputs)

      # add trace symbols
      trace_outputs = get_trace_outputs(mx.sym.Group(symbol_list))
      trace_size = len(trace_outputs)
      symbol_list.extend(trace_outputs)

      symbol = mx.sym.Group(symbol_list)
      executor = symbol.simple_bind(ctx=mx.cpu())
      add_variable_inputs(symbol, sym_input_dict, is_training=is_training)

      sym_names = symbol.list_arguments()
      xdl_inputs = []
      for sym in sym_names:
        xdl_inputs.append(sym_input_dict[sym])

      for aux in symbol.list_auxiliary_states():
        if aux in sym_input_dict:
          xdl_inputs.append(sym_input_dict[aux])
          sym_names.append(aux)

      target_size = len(executor.outputs)
      gradient_size=len(executor.grad_arrays)
      if device_type.lower() == 'cpu':
        outputs, gradients = xdl.mxnet_backend_op(
          inputs = xdl_inputs,
          var_name_str = ','.join(sym_names),
          device_type = device_type.lower(),
          graph_def=serialize_graph(symbol),
          target_size=target_size,
          gradient_size=gradient_size if is_training else 0,
          is_training=is_training,
          init_grad=init_grad if init_grad is not None else np.array([], dtype=np.float32),
          has_init_grad=True if init_grad is not None else False,
          id = random.randint(0, 2 ** 60))
      else:
        with xdl.device('GPU'):
          outputs, gradients = xdl.mxnet_backend_op(
            inputs = xdl_inputs,
            var_name_str = ','.join(sym_names),
            device_type = device_type.lower(),
            graph_def=serialize_graph(symbol),
            target_size=target_size,
            gradient_size=gradient_size if is_training else 0,
            is_training=is_training,
            init_grad=init_grad if init_grad is not None else np.array([], dtype=np.float32),
            has_init_grad=True if init_grad is not None else False,
            id = random.randint(0, 2 ** 60))

      # set trace outputs
      trace_outputs = [] if trace_size == 0 else outputs[-trace_size:]
      set_trace_outputs(sym_input_dict, trace_outputs)
      outputs = outputs if trace_size == 0 else outputs[:-trace_size]

      bn_var_num = len(bn_var_names)
      if bn_var_num > 0:
        bn_outputs = outputs[len(outputs) - bn_var_num:]
        outputs = outputs[0:len(outputs) - bn_var_num]
        bn_update_infos = zip(bn_var_names, bn_outputs, moments)
        add_to_collection(BN_STATISTIC, bn_update_infos)
        update_ops = []
        for n, v, m in bn_update_infos:
          update_op = xdl.ps_apply_moving_average_op(
            var_name = n, value = v, moment = m)
          update_ops.append(update_op)
        add_to_collection(UPDATE_OPS, update_ops)

      if is_training:
        sym_names_ = []
        gradients_ = []
        if gear_input_num > 0:
          global _GEAR_INPUTS
          gear_grads = [None] * gear_input_num
          for i in range(len(sym_names)):
            if sym_names[i] not in _GEAR_INPUTS:
              gradients_.append(gradients[i])
              sym_names_.append(sym_names[i])
            else:
              index = _GEAR_INPUTS.index(sym_names[i])
              gear_grads[index] = gradients[i]
          for i in range(len(gear_inputs)):
            set_gear_gradient(gear_inputs[i], gear_grads[i])
          add_to_collection(GEAR_GRAD, gear_grads, cur_model_scope())
          set_gradients(sym_names_, gradients_, cur_model_scope())
        else:
          set_gradients(sym_names, gradients, cur_model_scope())
      return outputs
    return _wrapper
  return decorator

def ams_main(main_fn, **tf_args):
  def _wrapper(*inputs, **kwargs):
    return mxnet_wrapper(**tf_args)(main_fn)(*inputs, **kwargs)
  return _wrapper

def ams_gear(forward_inputs, backward_inputs, init_grad, device_type='cpu'):
  def decorator(gear_fn):
    def _wrapper(*inputs, **kwargs):
      forwards = forward_inputs if isinstance(forward_inputs, list) else [forward_inputs]
      backwards = backward_inputs if isinstance(backward_inputs, list) else [backward_inputs]
      with xdl.model_scope("ams_gear_forward"):
        forward_results = mxnet_wrapper(is_training=False, device_type=device_type)(gear_fn)(
          *(forwards + list(inputs[1:])), **kwargs)
      with xdl.model_scope("ams_gear_backward"):
        _ = mxnet_wrapper(init_grad=init_grad, device_type=device_type)(gear_fn)(
          *(backwards + list(inputs[1:])), **kwargs)
      return forward_results
    return _wrapper
  return decorator

