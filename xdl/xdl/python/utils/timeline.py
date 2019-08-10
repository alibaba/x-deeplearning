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

import collections
import copy
import json
import re

from xdl.python.proto import perf_stats_pb2
from google.protobuf.text_format import Parse as TextParse
from xdl.python.utils.file_io import write_string_to_file

class ChromeTimeLineFormatter(object):
  def __init__(self, show_memory=False):
    self._show_memory = show_memory
    self._events = []
    self._metadata = []

  def _create_event(self, ph, category, name, pid, tid, timestamp):
    event = {}
    event['ph'] = ph
    event['cat'] = category
    event['name'] = name
    event['pid'] = pid
    event['tid'] = tid
    event['ts'] = timestamp
    return event

  def emit_pid(self, name, pid):
    event = {}
    event['name'] = 'process_name'
    event['ph'] = 'M'
    event['pid'] = pid
    event['args'] = {'name': name}
    self._metadata.append(event)

  def emit_tid(self, name, pid, tid):
    event = {}
    event['name'] = 'thread_name'
    event['ph'] = 'M'
    event['pid'] = pid
    event['tid'] = tid
    event['args'] = {'name': name}
    self._metadata.append(event)

  def emit_region(self, timestamp, duration, pid, tid, category, name, args):
    event = self._create_event('X', category, name, pid, tid, timestamp)
    event['dur'] = duration
    event['args'] = args
    self._events.append(event)

  def emit_obj_create(self, category, name, timestamp, pid, tid, object_id):
    event = self._create_event('N', category, name, pid, tid, timestamp)
    event['id'] = object_id
    self._events.append(event)

  def emit_obj_delete(self, category, name, timestamp, pid, tid, object_id):
    event = self._create_event('D', category, name, pid, tid, timestamp)
    event['id'] = object_id
    self._events.append(event)

  def emit_obj_snapshot(self, category, name, timestamp, pid, tid, object_id,
                        snapshot):
    event = self._create_event('O', category, name, pid, tid, timestamp)
    event['id'] = object_id
    event['args'] = {'snapshot': snapshot}
    self._events.append(event)

  def emit_flow_start(self, name, timestamp, pid, tid, flow_id):
    event = self._create_event('s', 'DataFlow', name, pid, tid, timestamp)
    event['id'] = flow_id
    self._events.append(event)

  def emit_flow_end(self, name, timestamp, pid, tid, flow_id):
    event = self._create_event('t', 'DataFlow', name, pid, tid, timestamp)
    event['id'] = flow_id
    self._events.append(event)

  def emit_counter(self, category, name, pid, timestamp, counter, value):
    event = self._create_event('C', category, name, pid, 0, timestamp)
    event['args'] = {counter: value}
    self._events.append(event)

  def emit_counters(self, category, name, pid, timestamp, counters):
    event = self._create_event('C', category, name, pid, 0, timestamp)
    event['args'] = counters.copy()
    self._events.append(event)

  def format_to_string(self, pretty=False):
    trace = {}
    trace['traceEvents'] = self._metadata + self._events
    if pretty:
      return json.dumps(trace, indent=4, separators=(',', ': '))
    else:
      return json.dumps(trace, separators=(',', ':'))

class Timeline(object):
  def __init__(self, perf_stats, graph=None):
    self._perf_stats = perf_stats_pb2.PerfStats()
    TextParse(perf_stats, self._perf_stats)    
    self._graph = graph
    self._formatter = ChromeTimeLineFormatter()
    self._tid_dict = {}
    self._next_tid = 0
    self._alloc_tids()

  def _alloc_tids(self):
    for node_stats in self._perf_stats.node_stats:
      if node_stats.node_name != '' and node_stats.op != '':
        tid = node_stats.thread_id
        if tid not in self._tid_dict:
          self._tid_dict[tid] = self._next_tid
          self._next_tid = self._next_tid + 1

  def _add_ops_stats(self):
    for node_stats in self._perf_stats.node_stats:
      if node_stats.node_name != '' and node_stats.op != '':
        self._add_op_stats(node_stats, pid=0)

  def _add_op_stats(self, nodestats, pid):
    node_name = nodestats.node_name
    start = nodestats.start_micros
    end = nodestats.end_micros
    duration = end - start
    tid = nodestats.thread_id
    op = nodestats.op
    args = {'name': node_name, 'op': op}
    self._formatter.emit_region(start, duration, pid, self._tid_dict[tid], 'Op', op, args)

  def to_string(self):
    self._add_ops_stats()
    return self._formatter.format_to_string(pretty=True)

  def save(self, path):
    write_string_to_file(path, self.to_string())

