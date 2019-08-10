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
import numpy as np
from xdl.python.framework.gradient import gradient
from xdl.python.framework.gradient import SparseGrad
from xdl.python.training.training_utils import get_global_step
from xdl.python.backend.model_scope import cur_model_scope
from xdl.python.sparse_engine.embedding import get_embedding_output, is_embedding_var
from xdl.python.training.gradient_utils import get_gradient, get_var_mapping
from xdl.python.framework.variable import trainable_variables
from xdl.python.utils.collections import *

class Optimizer(object):
  def __init__(self):
    self.gpu_ = None

  def optimize(self, var_list=None, update_global_step=True):
    if var_list == None:
      var_list = trainable_variables()
    sparse_var_grad = []
    update_ops = []
    shared_vars = set([])
    for var in var_list:
      grad_name = get_var_mapping(var)
      grad_name = grad_name if grad_name is not None else var.name
      grad = get_gradient(grad_name, cur_model_scope())
      if grad == None:
        print("[WARNING]: no gradient found for var:%s under scope:%s" %
              (var.name, cur_model_scope()), ", maybe not used?")
        continue

      if isinstance(grad, list):
        raise 'dupcate grad for var:', var
      if not is_embedding_var(var):
        update_ops.append(self.dense_update(var, grad))
      else:
        sparse_var_grad.append([var, grad])

    sparse_grads = self.compute_sparse_grad(sparse_var_grad)
    if len(sparse_grads) != len(sparse_var_grad):
      raise Exception("calc grad failed!")
    merged_sparse_grads = self.merge_sparse_grad(
      zip([x[0] for x in sparse_var_grad], sparse_grads))

    if get_collection("sparse_grad") == None:
      add_to_collection("sparse_grad", {})
    sparse_grad_dict = get_collection("sparse_grad")[0]
    for i in range(len(merged_sparse_grads)):
      if not isinstance(merged_sparse_grads[i][1], SparseGrad):
        raise Exception("embedding var must hava sparse grads")
      sparse_grad_dict[merged_sparse_grads[i][0].name] = merged_sparse_grads[i][1]
      update_ops.append(self.sparse_update(
          merged_sparse_grads[i][0],
          merged_sparse_grads[i][1].grad,
          merged_sparse_grads[i][1].indices))
    if update_global_step:
      update_ops.append(self.update_global_step_op())
    return update_ops

  def merge_sparse_grad(self, sparse_var_grads):
    sparse_var_dict = {}
    merged_sparse_grads = []
    name_2_var = {}
    for var, grad in sparse_var_grads:
      if var.name not in sparse_var_dict:
        sparse_var_dict[var.name] = []
      sparse_var_dict[var.name].append(grad)
      name_2_var[var.name] = var
    for var, grads in sparse_var_dict.items():
      merged_sparse_grads.append(
        (name_2_var[var], grads[0] if len(grads) == 1 else self._add_sparse_grads(grads)))
    return merged_sparse_grads

  def _add_sparse_grads(self, grads):
    id_list = []
    grad_list = []
    for grad in grads:
      id_list.append(grad.indices)
      grad_list.append(grad.grad)
    grad, indices = xdl.sparse_grad_add_op(in_grads=grad_list, in_ids=id_list, size=len(id_list))
    return SparseGrad(grad, indices)

  def update_global_step_op(self):
    global_step = get_global_step()
    update_op = xdl.ps_assign_add_op(
      var_name = global_step.name,
      var_type = global_step.vtype,
      delta = np.array(1, dtype=np.int64))
    return update_op

  def compute_sparse_grad(self, sparse_var_grad):
    inputs = []
    outputs = []
    in_grads = {}
    for var, grad in sparse_var_grad:
      inputs.append(var.grad_tensor)
      outputs.append(get_embedding_output(var))
      if outputs[-1] is None:
        raise Exception('embedding output is None for var:', var.name)
      in_grads[outputs[-1]] = grad
    backend_device_type = get_collection(BACKEND_DEVICE_TYPE)[0]
    return gradient(inputs, outputs, in_grads)
    #if backend_device_type == 'gpu':
    #  with xdl.device('GPU'):
    #    return gradient(inputs, outputs, in_grads)
    #else:
    #  with xdl.device('CPU'):
    #    return gradient(inputs, outputs, in_grads)

  def dense_update(self, var, grad):
    """update dense gradient to ps
    Args:
    var: xdl dense variable
    grad: gradient tensor
    Returns:
    a dense_update op
    """
    raise Exception("unemplement")

  def sparse_update(self, var, grad, indices):
    """update sparse gradient to ps
    Args:
    var: xdl sparse variable
    grad: dense gradients
    indices: sparse index for gradients
    Returns:
    a sparse_update op
    """
    raise Exception("unemplement")

