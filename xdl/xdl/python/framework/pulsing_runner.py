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

from xdl.python.lib.graph import current_graph, Graph
from xdl.python.lib.tensor import Tensor
from collections import defaultdict
from xdl.python.lib.gen_attr import gen_int, gen_type, DataType
from xdl.python import pybind
from xdl.python.utils.timeline import Timeline

def _decode_input(name):
  if name[0] == '^':
    return name[1:], -1
  else:
    x = name.find(':')
    return name[:x], int(name[x+1:])

def _flatten_outs(outs):
  if isinstance(outs, Tensor):
    return [outs.define]
  elif isinstance(outs, (list, tuple, set)):
    return sum([_flatten_outs(i) for i in outs], [])
  elif isinstance(outs, dict):
    return sum([_flatten_outs(i) for i in outs.values()], [])
  else:
    raise ValueError("Unknown Type: " + str(outs.__class__))

class PulsingRunner(object):
  def __init__(self, outs, g = None, weak_variable = False, slow_start = 1000):
    if g is None:
      g = current_graph()
    self._g = g
    self._outs = _flatten_outs(outs)
    self._real_outs = []
    self._weak_variable = weak_variable
    self._slow_start = slow_start
    self._step = False
    self._build()
    self._run_time = 0

  def step(self, perf = None):
    if self._slow_start > 0:
      self._slow_start -= 1
      self._run_time += 1
      if perf:
        run_option = pybind.RunOption()
        run_option.perf = True
        run_statistic = pybind.RunStatistic()
        rst = self._g.execute(self._outs, run_option=run_option, run_statistic=run_statistic)
        Timeline(run_statistic.perf_result).save(str(perf))
      else:
        rst = self._g.execute(self._outs)
      return rst
    if self._step is False:
      self._step = True
      self._run_ops = []
      for i in self._level_nums[:-1]:
        run_option = pybind.RunOption()
        new_ctx = pybind.ExecutorContext(self._channel_count)
        run_option.set_in_ctx(self._ctx)
        run_option.set_out_ctx(new_ctx)
        self._run_ops.extend(self._level_run[i])
        self._new_graph[i].execute(self._run_ops, run_option=run_option)
        self._ctx = new_ctx
      self._run_ops = self._real_outs + ["^" + self._dst_node.name]
    run_option = pybind.RunOption()
    new_ctx = pybind.ExecutorContext(self._channel_count)
    run_option.set_in_ctx(self._ctx)
    run_option.set_out_ctx(new_ctx)
    # timeline start
    self._run_time += 1
    if perf:
      run_option.perf = True
      run_statistic = pybind.RunStatistic()
      rst = self._dst_graph.execute(self._run_ops, run_option=run_option, run_statistic=run_statistic)[:-1]
      Timeline(run_statistic.perf_result).save(str(perf))
    else:
      rst = self._dst_graph.execute(self._run_ops, run_option=run_option)[:-1]
    self._ctx = new_ctx
    return rst

  def _build(self):
    self._level_nodes = defaultdict(lambda:[])
    levels = self._g.levels()
    nodes = self._g.nodes()
    self._nodes, self._levels = self._prune_node(self._outs, nodes, levels)
    for k, v in self._levels.items():
      self._level_nodes[v].append(k)
    self._level_nums = list(self._level_nodes.keys())
    self._level_nums.sort()
    self._level_id = dict(zip(self._level_nums, range(len(self._level_nums))))
    self._level_channel = {k:{} for k in self._level_nums}
    self._level_run = {k:set() for k in self._level_nums}
    self._channel_count = 0
    self._new_nodes = []
    self._level_nodes = defaultdict(lambda:[])
    self._xnodes = {}
    self._ctx = pybind.ExecutorContext(self._channel_count)

    self._process_nodes()
    self._process_outs()
    
    self._new_graph = {}
    for level in self._level_nums:
      self._add_var_dependencies(level)
      self._new_graph[level] = self._build_graph(level)

    self._add_channel_dst()
    self._dst_graph = self._build_all_graph()

  def _process_nodes(self):
    for k, v in self._levels.items():
      self._process_node(self._nodes[k], v)

  def _process_outs(self):
    level = self._level_nums[-1]
    for i in self._outs:
      out = self._get_output(i, level, "output")
      if out is not None:
        self._real_outs.append(out)

  def _build_graph(self, level):
    ret = Graph()
    for l in self._level_nums:
      if l > level:
        break
      for node in self._level_nodes[l]:
        ret.add_node_internal(pybind.NodeDef(node), 0)
    return ret

  def _build_all_graph(self):
    ret = Graph()
    for node in self._xnodes.values():
      ret.add_node_internal(pybind.NodeDef(node), 0)
    return ret

  def _prune_node(self, outs, nodes, levels):
    dfs_list = [_decode_input(i)[0] for i in outs]
    dfs_set = set(dfs_list)
    p = 0
    while p < len(dfs_list):
      cur_node = dfs_list[p]
      p += 1
      inputs = [_decode_input(i)[0] for i in nodes[cur_node].input]
      for i in inputs:
        if i not in dfs_set:
          dfs_set.add(i)
          dfs_list.append(i)

    nodes = {k:v for k, v in nodes.items() if k in dfs_set}
    levels = {k:v for k, v in levels.items() if k in dfs_set}
    return nodes, levels

  def _process_node(self, node, level):
    new_node = self._add_node(node.name, level)
    new_node.op = node.op
    for inx in node.input:
      iny = self._get_output(inx, level, node.name)
      if iny is not None:
        new_node.input.append(iny)
    for ot in node.output_type:
      new_node.output_type.append(ot)
    for k, v in node.attr.items():
      new_node.attr[k] = v
    new_node.device = node.device

  def _get_output(self, spec, level, from_node):
    node, idx = _decode_input(spec)
    if (self._levels[node] > level):
      raise ValueError("Cannot Get Output {} at Level {}, from node {}".format(spec, level, from_node))
    if (self._levels[node] == level):
      return spec
    if idx == -1:
      self._level_run[self._levels[node]].add(spec)
      return None
    else:
      x = spec
      for i in range(self._level_id[self._levels[node]], self._level_id[level]):
        lin = self._level_nums[i]
        lout = self._level_nums[i + 1]
        if spec in self._level_channel[lin]:
          x = self._level_channel[lin][spec][1]
        else:
          self._level_channel[lin][spec] = self._add_channel(x, lin, lout)
          x = self._level_channel[lin][spec][1]
          self._level_run[lin].add(self._level_channel[lin][spec][0])
      return x

  def _add_channel(self, x_node, src_level, dst_level):
    node_name = _decode_input(x_node)[0]
    if node_name in self._nodes:
      node_def = self._nodes[node_name]
    else:
      node_def = self._xnodes[node_name]
    node_type = node_def.output_type[_decode_input(x_node)[1]]
    channel = self._channel_count
    in_node = self._add_node("/pulsing/channel{}/in".format(channel), src_level)
    out_node = self._add_node("/pulsing/channel{}/out".format(channel), dst_level)
    self._channel_count += 1
    in_node.op = "WriteContextOp"
    in_node.input.append(x_node)
    in_node.attr["ctx_id"] = gen_int(channel, "ctx_id")
    in_node.attr["dtype"] = gen_type(node_type, "dtype")
    in_node.device = node_def.device
    out_node.op = "ReadContextOp"
    out_node.attr["ctx_id"] = gen_int(channel, "ctx_id")
    out_node.attr["dtype"] = gen_type(node_type, "dtype")
    out_node.output_type.append(node_type)
    out_node.device = node_def.device
    return "^/pulsing/channel{}/in".format(channel), "/pulsing/channel{}/out:0".format(channel)

  def _add_node(self, name, level):
    node = pybind.NodeDef()
    node.name = name
    self._new_nodes.append(node)
    self._level_nodes[level].append(node)
    self._xnodes[name] = node
    return node

  def _add_var_dependencies(self, level):
    src_nodes = defaultdict(lambda:[])
    dst_nodes = defaultdict(lambda:[])
    for node_name in self._nodes:
      node = self._xnodes[node_name]
      node_level = self._levels[node_name]
      nm = node.name
      op = node.op
      if op[:2] == "Ps" and op[-2:] == "Op":
        var_name = node.attr["var_name"].s
        if node_level == level:
          src_nodes[var_name].append(node)
        elif node_level < level:
          dst_nodes[var_name].append(node)
    for var_name in src_nodes:
      src_name = ["^" + i.name for i in src_nodes[var_name]]
      for node in dst_nodes[var_name]:
        for namex in src_name:
          node.input.append(namex)

    for node_name in self._nodes:
      node = self._xnodes[node_name]
      process = True
      for i in node.input:
        pos = i.find("/pulsing/channel")
        if pos != 0 and pos != 1:
          process = False
          break
      if process:
        level_id = self._level_id[level]
        if level_id != 0:
          xlevel = self._level_nums[level_id - 1]
          for channel in self._level_channel[xlevel].values():
            node.input.append("^" + channel[1][:-2])

  def _add_channel_dst(self):
    self._dst_node = self._add_node("/pulsing/dst", None)
    self._dst_node.op = "NoOp"
    self._dst_node.device.device_name = "CPU"
    for level in self._level_nums:
      for dst in self._level_run[level]:
        self._dst_node.input.append("^" + _decode_input(dst)[0])
