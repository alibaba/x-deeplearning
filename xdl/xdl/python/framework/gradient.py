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

from collections import defaultdict
from xdl.python.lib.graph import current_graph, device
from xdl.python.utils.collections import get_collection

class SparseGrad(object):
  def __init__(self, grad, indices):
    self._grad = grad
    self._indices = indices

  @property
  def grad(self):
    return self._grad

  @property
  def indices(self):
    return self._indices

class GradientManager(object):
  _current_manager = None

  def __init__(self):
    self._gradient_map = {}
    self._default_gradient = None

  def def_gradient(self, name):
    def wrapper(func):
      if name in self._gradient_map:
        raise ValueError("Gradient Multi Define")
      self._gradient_map[name] = GradientCalc.simple_gradient_wrapper(func)
      return func
    return wrapper

  def def_gradient_internal(self, name):
    def wrapper(func):
      if name in self._gradient_map:
        raise ValueError("Gradient Multi Define")
      self._gradient_map[name] = func
      return func
    return wrapper

  def def_default_gradient(self):
    def wrapper(func):
      if self._default_gradient is not None:
        raise ValueError("Default Gradient Multi Define")
      self._default_gradient = func
      return func
    return wrapper

  def gradient_func(self, name):
    if name in self._gradient_map:
      return self._gradient_map[name]
    else:
      return None

  def default_gradient_func(self):
    return self._default_gradient

  def gradient(self, inputs, outputs, output_gradients = None):
    calc = GradientCalc(self, inputs, outputs, output_gradients)
    calc.run()
    return calc.result()

  @staticmethod
  def current_manager():
    return GradientManager._current_manager

GradientManager._current_manager = GradientManager()

class GradientCalc(object):
  def __init__(self, manager, inputs, outputs, output_gradients):
    self._manager = manager
    self._inputs = inputs
    self._outputs = outputs
    self._output_gradients = output_gradients
    self._output_list = defaultdict(lambda:[])
    self._gradients = {}
    self._op_gradients = {}
    self._result = []

  def run(self):
    for i in self._outputs:
      if self._output_gradients is None or i not in self._output_gradients:
        self._gradients[i] = self._manager.default_gradient_func()()
      else:
        self._gradients[i] = self._output_gradients[i]

    self.feed_output_list()

    for i in self._inputs:
      self._result.append(self.get_gradient(i))

  def result(self):
    return list(self._result)

  def feed_output_list(self):
    ops = current_graph().ops()
    for op in ops.values():
      for i in range(len(op.inputs)):
        self._output_list[op.inputs[i]].append((op, i))

  def get_op_gradient(self, op):
    if op not in self._op_gradients:
      func = self._manager.gradient_func(op.op)
      if func is None:
        self._op_gradients[op] = None
      else:
        grads = [self.get_gradient(i) for i in op.outputs]
        has_grad = False
        for grad in grads:
          if grad is not None:
            has_grad = True
            break
        if has_grad:
          with device(op.device_name):
            self._op_gradients[op] = func(op, self.get_gradient)
        else:
          self._op_gradients[op] = None
    return self._op_gradients[op]

  def calc_gradient(self, node):
    grad = None
    for op, i in self._output_list[node]:
      op_grad = self.get_op_gradient(op)
      if op_grad is None:
        continue
      if op_grad[i] is not None:
        if grad is not None:
          grad = grad + op_grad[i]
        else:
          grad = op_grad[i]
    return grad

  def get_gradient(self, node):
    if node not in self._gradients:
      self._gradients[node] = self.calc_gradient(node)
    return self._gradients[node]

  @staticmethod
  def simple_gradient_wrapper(func):
    def wrapper(op, getter):
      output_grad = [getter(i) for i in op.outputs]
      return func(op, output_grad)
    return wrapper

def def_gradient(name):
  return GradientManager.current_manager().def_gradient(name)

def def_gradient_internal(name):
  return GradientManager.current_manager().def_gradient_internal(name)

def def_default_gradient():
  return GradientManager.current_manager().def_default_gradient()

def gradient(inputs, outputs, output_gradients = None):
  return GradientManager.current_manager().gradient(inputs, outputs, output_gradients)

def get_sparse_grads(name):
  return get_collection('sparse_grad')[0][name]
