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

import random
import contextlib
import numpy
from xdl.python import pybind
from xdl.python.lib.tensor import Tensor
from xdl.python.lib.op import Op
from xdl.python.lib.gen_attr import gen_attr
from xdl.python.lib.error import check_error

class Graph(object):
  _current_graph = []

  @staticmethod
  def current_graph():
    return Graph._current_graph[-1]

  @staticmethod
  def default_device():
    ret = pybind.DeviceDef()
    ret.device_name = "CPU"
    return ret

  def __init__(self):
    self._graph_def = pybind.GraphDef()
    self._graph_def.hash = random.randint(0, 2 ** 60)
    self._device = Graph.default_device()
    self._ops = {}
    self._nodes = {}
    self._namescope = '/'
    self._control_dependencies = []
    self._level = 0
    self._levels = {}
    self._finalize = False

  def __enter__(self):
    Graph._current_graph.append(self)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    Graph._current_graph.pop()

  def finalize(self):
    self._finalize = True

  def unfinalize(self):
    self._finalize = False

  @contextlib.contextmanager
  def device(self, device_name, **kwargs):
    try:
      old_device = self._device
      ret = pybind.DeviceDef()
      ret.device_name = device_name.upper()
      for k, v in kwargs.items():
        ret.attr[k] = str(v)
      self._device = ret
      yield
    finally:
      self._device = old_device

  @contextlib.contextmanager
  def namescope(self, name):
    try:
      old_namescope = self._namescope
      if name[-1] != '/':
        name = name + '/'
      if name[0] == '/':
        self._namescope = name
      else:
        self._namescope = self._namescope + name
      yield self._namescope
    finally:
      self._namescope = old_namescope

  @contextlib.contextmanager
  def control_dependencies(self, deps):
    try:
      old_cotrol_dependencies = self._control_dependencies[:]
      if deps is None:
        self._control_dependencies = []
      else:
        if isinstance(deps, list):
          self._control_dependencies += deps
        else:
          self._control_dependencies.append(deps)
      yield
    finally:
      self._control_dependencies = old_cotrol_dependencies

  @contextlib.contextmanager
  def level(self, l):
    try:
      old_level = self._level
      self._level = l
      yield
    finally:
      self._level = old_level

  def _create_name(self, name, name_hint):
    if name is None:
      k = 0
      while (True):
        if k == 0:
          real_name = name_hint
        else:
          real_name = name_hint + "_" + str(k)
        k += 1
        real_name = self._namescope + real_name
        if not real_name in self._nodes:
          return real_name
    elif name[0] == '/':
      return name
    else:
      return self._namescope + name

  def add_node(self, op_def, name, inputs, attrs, output_types):
    if self._finalize:
      raise Exception("graph has been finalized")
    name = self._create_name(name, op_def.name)
    if name in self._nodes:
      raise Value("Duplicate Define Op " + name)
    real_inputs = []
    for i in inputs:
      real_inputs.append(i.define)
    for i in self._control_dependencies:
      real_inputs.append(i.op.depend().define)
    attr_hint = {}
    real_attrs = {}
    for attr in op_def.attrs:
      attr_hint[attr.name] = attr.type
    for k, v in attrs.items():
      real_attrs[k] = gen_attr(v, k, attr_hint[k])
    node = pybind.NodeDef()
    node.name = name
    node.op = op_def.name
    for i in output_types:
      node.output_type.append(i)
    for i in real_inputs:
      node.input.append(i)
    node.device = self._device
    for k, v in real_attrs.items():
      node.attr[k] = v
    self.add_node_internal(node, self._level)
    op = Op(inputs, attrs, name, op_def.name, node.device.device_name)
    self._ops[name] = op
    return op

  def add_node_internal(self, node, level):
    self._graph_def.node.append(node)
    self._nodes[node.name] = node
    self._levels[node.name] = level

  def ops(self):
    return dict(self._ops)

  def nodes(self):
    return dict(self._nodes)

  def levels(self):
    return dict(self._levels)

  def execute(self, outputs, run_option=None, run_statistic=None):
    if run_option and run_option.perf:
      if run_statistic is None:
        raise 'run_statistic must be specified when perf is turned on'
    output_define = []
    def recursive_feed_output(x, k):
      if isinstance(x, Tensor):
        x = x.define
      if isinstance(x, (str, unicode)):
        output_define.append(x)
        if x[0] == '^':
          return None, k
        else:
          return k, k + 1
      elif isinstance(x, (list, tuple, set)):
        rst = []
        for i in x:
          y, k = recursive_feed_output(i, k)
          rst += [y]
        return x.__class__(rst), k
      elif isinstance(x, dict):
        rst = {}
        for i in x:
          y, k = recursive_feed_output(x[i], k)
          rst[i] = y
        return rst, k
      else:
        raise ValueError("cannot execute type {}".format(x))
    output_spec, _ = recursive_feed_output(outputs, 0)
    xdl_output_spec = pybind.OutputSpec();
    xdl_output_spec.output = pybind.StringVector(output_define)
    xdl_output_spec.output_device = Graph.default_device()
    run_option = run_option if run_option is not None else pybind.RunOption()
    result = pybind.execute(self._graph_def, xdl_output_spec, run_option)
    check_error(result.status)
    outputs = result.outputs
    if run_option and run_option.perf:
      run_statistic.perf_result = result.run_statistic.perf_result;
    def recursive_build_result(x):
      if x is None:
        return None
      if isinstance(x, (int, long)):
        return numpy.array(outputs[x], copy = False)
      elif isinstance(x, (list, tuple, set)):
        rst = []
        for i in x:
          y = recursive_build_result(i)
          rst += [y]
        return x.__class__(rst)
      elif isinstance(x, dict):
        rst = {}
        for i in x:
          y = recursive_build_result(x[i])
          rst[i] = y
        return rst
      else:
        raise ValueError("Internal Error")
    return recursive_build_result(output_spec)

  def execute_loop(self, *outputs):
    def recursive_feed_output(x, output_define):
      if isinstance(x, Tensor):
        output_define.append(x.define)
      elif isinstance(x, (list, tuple, set)):
        for i in x:
          recursive_feed_output(i, output_define)
      elif isinstance(x, dict):
        for i in x:
          recursive_feed_output(x[i], output_define)
      else:
        raise ValueError("cannot execute type {}".format(x))
    if (len(outputs) == 0):
      raise ValueError("execute loop need at least one state")
    output_spec = pybind.OutputSpecVector()
    for o in outputs:
      spec = []
      recursive_feed_output(o, spec)
      xdl_spec = pybind.OutputSpec()
      xdl_spec.output = pybind.StringVector(spec)
      xdl_spec.output_device = Graph.default_device()
      output_spec.append(xdl_spec)
    pybind.execute_loop(self._graph_def, output_spec)

Graph._current_graph.append(Graph())

def current_graph():
  return Graph.current_graph()

def namescope(name):
  return current_graph().namescope(name)

def device(name, **kwargs):
  return current_graph().device(name, **kwargs)

def control_dependencies(deps):
  return current_graph().control_dependencies(deps)

def level(l):
  return current_graph().level(l)

def execute(outputs, run_option=None, run_statistic=None):
  return current_graph().execute(outputs, run_option, run_statistic)

def execute_loop(*outputs):
  return current_graph().execute_loop(*outputs)

def execute_loop_wait():
  check_error(pybind.execute_loop_wait())

def create_op(op_def, name, inputs, attrs, output_spec):
  def _recursive_list(x):
    if isinstance(x, (list, tuple)):
      return sum([_recursive_list(i) for i in x], [])
    else:
      return [x]
  ret = current_graph().add_node(op_def, name, _recursive_list(inputs), attrs, _recursive_list(output_spec))
  def _recursive_create_output(x, k):
    if isinstance(x, (list, tuple)):
      l = []
      for i in x:
        k, v = _recursive_create_output(i, k)
        l.append(v)
      return k, l
    else:
      return k + 1, Tensor(ret.output_define(k), x, ret)
  ret.set_outputs(_recursive_create_output(output_spec, 0)[1])
  return ret
