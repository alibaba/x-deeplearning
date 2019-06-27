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
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.ops import variables
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.framework import tensor_shape
from tensorflow.python.saved_model import utils as tensor_utils

import xdl
from xdl.python.backend.tf.convert_utils import XDL2TF
from xdl.python.training.gradient_utils import set_gradients, set_gear_gradient
from xdl.python.training.gradient_utils import set_gradients, add_var_mapping
from xdl.python.lib.tensor import Tensor
from xdl.python.utils.collections import *
from xdl.python.sparse_engine.embedding import is_embedding_var, get_embedding_info
from xdl.python.backend.model_scope import cur_model_scope
import xdl.python.backend.tf.tf_hook
from xdl.python.backend.tf import tf_batchnorm_hook

"""python adapter for tensorflow."""

def recursive_make_placeholder(x, xdl_inputs, tf_inputs):
  if isinstance(x, (tuple, list)):
    return [recursive_make_placeholder(item, xdl_inputs, tf_inputs) for item in x]
  elif isinstance(x, Tensor):
    placeholder = make_placeholder(x)
    xdl_inputs.append(x)
    tf_inputs.append(placeholder)
    return placeholder
  else:
    return x

def make_placeholder(x):
  """define a tensorflow placeholder for xdl input x.
  Args:
    x: a xdl dense or embedding tensor
    Returns:
    a tf placeholder
  Raises:
    None
  """
  emb_info = get_embedding_info(x)
  if emb_info is not None:
    placeholder = tf.placeholder(
      tf.float32,
      name=emb_info.name, 
      shape=[None, emb_info.emb_dim])
    emb_info._output_tensor = x
    add_var_mapping(emb_info.var, placeholder.name)
    add_to_collection(BACKPROP_VARS, (placeholder.name, placeholder))        
    return placeholder
  else:
    if x.shape is not None and len(x.shape) > 1:
      return tf.placeholder(XDL2TF.convert_type(x.dtype), shape=[None] + list(x.shape[1:]))
    else:
      return tf.placeholder(XDL2TF.convert_type(x.dtype), shape=x.shape)      

def serialize_graph(clear_devices=False, as_text=False):
  """serialize tf graph to path."""
  saver = tf_saver.Saver(
    variables._all_saveable_objects(),
    sharded=True,
    write_version=saver_pb2.SaverDef.V2,
    allow_empty=True)
  meta_graph_def = saver.export_meta_graph(clear_devices=clear_devices)
  if as_text:
    return str(meta_graph_def)
  else:
    return meta_graph_def.SerializeToString()

def get_op_name(op):
  """get tf op name in op_list"""
  return tensor_utils.build_tensor_info(op).name

def get_op_names(op_list):
  """get tf op name in op_list"""
  flatten_op_list = flatten_list(op_list)
  return [get_op_name(x) for x in flatten_op_list]

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

def add_backprop_ops(loss, backprop_collection, init_grad=None):
  """add backprop ops for vars in tf-graph."""
  backprop_vars = [x[1] for x in backprop_collection]
  grads = tf.gradients(loss, backprop_vars, grad_ys=init_grad)
  var_names = []
  grad_op_names = []
  for i in range(len(grads)):
    if grads[i] is not None:
      var_name = backprop_collection[i][0]
      grad_op = tf.identity(grads[i])
      var_names.append(var_name)
      grad_op_names.append(get_op_name(grad_op))
  return var_names, grad_op_names

def add_variable_inputs(inputs, input_op_names):
  """deal inputs for variables defined in tf-graph"""
  var_mapping = get_collection(VAR_MAPPING, cur_model_scope())
  if var_mapping is None:
    return
  inputs.extend([x[0].value for x in var_mapping])
  input_op_names.extend([x[1].name for x in var_mapping])

def tf_wrapper(is_training=True, init_grad=None, gpu_memory_fraction=0.5, device_type='cpu'):
  """python decorator to adapt a tf-model define function to xdl.
  
  Args:
  model_func: a tf-model define function, the first return value must be loss

  Returns:
  a list of xdl tensors returned by model_func

  Raises:
  None
  """
  def decorator(model_func):
    def _wrapper(*inputs, **kwargs):
      add_to_collection(BACKEND_DEVICE_TYPE, device_type.lower())
      model_fn_inputs = []
      xdl_inputs = []
      placeholders = []

      for x in inputs:
        input = recursive_make_placeholder(x, xdl_inputs, placeholders)
        model_fn_inputs.append(input)

      gear_placeholders = []
      if 'gear_inputs' in kwargs:
        gear_inputs = kwargs['gear_inputs']
        input = recursive_make_placeholder(gear_inputs, xdl_inputs, placeholders)
        gear_placeholders = flatten(placeholders[-len(gear_inputs):])
        kwargs['gear_inputs'] = input

      init_grad_placeholder = None
      if init_grad is not None:
        init_grad_placeholder = recursive_make_placeholder(
          init_grad, xdl_inputs, placeholders)

      targets = model_func(*model_fn_inputs, **kwargs)
      local_init_op_names = [x.initializer.name for x in tf.local_variables()]
      if isinstance(targets, tuple):
        targets = list(targets)
      else:
        targets = [targets]
      # add batch_normalization
      batchnorm_begin = len(targets)
      batchnorm_tensors = tf_batchnorm_hook.get_batchnorm_tensors()
      batchnorm_size = len(batchnorm_tensors)
      targets.extend(batchnorm_tensors)

      var_names = []
      gradient_op_names = []
      if is_training:
        loss = targets[0]
        if isinstance(loss, (list, tuple, dict)):
          raise 'model function must reture loss as first output'
        for gear_placeholder in gear_placeholders:
          add_to_collection(BACKPROP_VARS, ("gear_grad", gear_placeholder))        
        var_names, gradient_op_names = add_backprop_ops(
          loss,
          get_collection(BACKPROP_VARS, ['', cur_model_scope()]),
          init_grad_placeholder)
      input_op_names = get_op_names(placeholders)
      target_op_names = get_op_names(targets)
      op_inputs = xdl_inputs
      add_variable_inputs(op_inputs, input_op_names)
      outputs, gradients = xdl.tfbackend_op(
        inputs = list(op_inputs),
        input_op_names = ','.join(input_op_names),
        target_op_names = ','.join(target_op_names),
        gradient_op_names = ','.join(gradient_op_names),
        local_init_op_names = ','.join(local_init_op_names),
        graph_def=serialize_graph(),
        target_size=len(target_op_names),
        gradient_size=len(gradient_op_names),
        gpu_memory_fraction=gpu_memory_fraction)

      gradients_size = len(gradients)
      gear_size = len(gear_placeholders)
      gear_grads = gradients[gradients_size - gear_size:]
      gradients = gradients[0: gradients_size - gear_size]
      var_names = var_names[0: gradients_size - gear_size]
      if len(gear_grads) > 0:
        add_to_collection(GEAR_GRAD, gear_grads, cur_model_scope())
        for i in range(len(gear_inputs)):
          set_gear_gradient(gear_inputs[i], gear_grads[i])
      if is_training:
        set_gradients(var_names, gradients, cur_model_scope())
      if batchnorm_size == 0:
        batchnorm_output = []
      else:
        batchnorm_output = outputs[-batchnorm_size:]
      tf_batchnorm_hook.set_tf_output(batchnorm_output)
      return outputs if batchnorm_size == 0 else outputs[:-batchnorm_size]
    return _wrapper
  return decorator

def ams_main(main_fn, **tf_args):
  def _wrapper(*inputs, **kwargs):
    return tf_wrapper(**tf_args)(main_fn)(*inputs, **kwargs)
  return _wrapper

def ams_gear(forward_inputs, backward_inputs, init_grad, **tf_args):
  def decorator(gear_fn):
    def _wrapper(*inputs, **kwargs):
      forwards = forward_inputs if isinstance(forward_inputs, list) else [forward_inputs]
      backwards = backward_inputs if isinstance(backward_inputs, list) else [backward_inputs]
      with xdl.model_scope("ams_gear_forward"):
        with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
          forward_results = tf_wrapper(is_training=False, **tf_args)(gear_fn)(*(forwards + list(inputs[1:])), **kwargs)
      with xdl.model_scope("ams_gear_backward"):
        with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
          _ = tf_wrapper(init_grad=init_grad, **tf_args)(gear_fn)(*(backwards + list(inputs[1:])), **kwargs)
      return forward_results
    return _wrapper
  return decorator
        
        
    
