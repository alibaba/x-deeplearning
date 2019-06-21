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

import xdl
import os
import numpy as np
from xdl.python.lib.graph import current_graph
from xdl.python.proto import graph_def_pb2
from google.protobuf.text_format import Parse as TextParse
from xdl.python.utils.file_io import write_string_to_file
from xdl.python.framework.session import Hook
from xdl.python.training.training_utils import get_global_step
from xdl.python.lib.graph import execute
from xdl.python.utils.config import get_ckpt_dir
from xdl.python.io.data_io import DataIO

def _string_to_int8(src):
    return np.array([ord(ch) for ch in src], dtype=np.int8)

def _graphdef_to_pb(graph_def):
    pb = graph_def_pb2.GraphDef()
    TextParse(graph_def.to_proto_string(), pb)
    return pb

'''
Usage:
    xdl.graph_tag().set_dataio_input(data_io)
    xdl.graph_tag().set_output(prop) 
'''
class GraphTag(object):
    def __init__(self):
        self._inputs = list()
        self._output_op_names = list()
    def append_input(self, op_name, input_name, type, size=1, table=0):
        if type == xdl.features.sparse:
            input_type = graph_def_pb2.kSparse
        else:
            input_type = graph_def_pb2.kDense
        self._inputs.append((op_name, input_name, input_type, size, table))
    def set_dataio_input(self, data_io):
        if not isinstance(data_io, DataIO):
            raise ValueError('set_dataio_input only accept data_io as parameter')
        for (idx, name, type, nvec, table) in data_io.tags:
            if type == xdl.features.sparse:
                input_type = graph_def_pb2.kSparse
            else:
                input_type = graph_def_pb2.kDense
            self._inputs.append(('/GetBatch:%d' % idx, name, input_type, nvec, table))
    def set_input(self, op, tag_name):
        self._inputs.append((op.define, tag_name, graph_def_pb2.kDense, 0, 0))
    def set_output(self, op):
        self._output_op_names.append(op.define)
    def set_mx_output(self, backend_symbol):
        import mxnet
        if isinstance(backend_symbol, mxnet.symbol.symbol.Symbol):
            self._output_op_name = backend_symbol.name
    def set_tf_output(self, backend_symbol):
        import tensorflow
        if isinstance(backend_symbol, tensorflow.Tensor):
            self._output_op_name = backend_symbol.name

    @property
    def inputs(self):
        return self._inputs

    @property
    def output_op_names(self):
        return self._output_op_names

_GRAPH_TAG = GraphTag()

def graph_tag():
    global _GRAPH_TAG
    return _GRAPH_TAG

class Saver(object):
    def __init__(self):
        self._ckpt_dir = xdl.get_ckpt_dir()
        if self._ckpt_dir is None:
            raise ValueError('must specify ckpt_dir arg in cmdline')
        self._graph_def = _graphdef_to_pb(current_graph()._graph_def)
    def save(self, version):
        execute(self.save_op(version))
    def restore(self, version):
        execute(self.restore_op(version))
    def save_op(self, version):
        return xdl.ps_save_op(_string_to_int8(version))
    def restore_op(self, version):
        return xdl.ps_restore_op(_string_to_int8(version))
    def tag_reader_input(self, reader):
        graph_tag().set_dataio_input(reader)
    def tag_input(self, op, tag_name):
        graph_tag().set_input(op, tag_name)
    def tag_output(self, op):
        if isinstance(op, list):
            for item in op:
                graph_tag().set_output(item)
        else:
            graph_tag().set_output(op)            
    def export_graph(self, as_text=True):
        for (op_name, input_name, input_type, size, table) in graph_tag().inputs:
            self.append_input(op_name, input_name, input_type, size, table)
        for output in graph_tag().output_op_names:
            output_tag = self._graph_def.tag.output.add()
            output_tag.op_name = output
        if as_text:
            path = os.path.join(self._ckpt_dir, "graph.txt")
            write_string_to_file(path, str(self._graph_def))
        else:
            path = os.path.join(self._ckpt_dir, "graph.pb")
            write_string_to_file(path, self._graph_def.SerializeToString())
    def append_input(self, op_name, input_name, input_type, size=1, table=0):
        inp = self._graph_def.tag.input.add()
        inp.op_name = op_name
        inp.input_name = input_name
        inp.input_type = input_type
        inp.size = size
        inp.table = table

class CheckpointHook(Hook):
    def __init__(self, save_interval_step, is_training=True, export_graph=True, as_text=True):
        super(CheckpointHook, self).__init__()
        self._global_step = get_global_step()
        self._save_interval = save_interval_step
        self._ckpt_dir = get_ckpt_dir()
        self._saver = Saver()
        self._is_training = is_training
        self._save_cnt = 0
        self._first_run = True
        self._export_graph = export_graph
        self._as_text = as_text

    def before_run(self, v):
        if self._ckpt_dir is None:
            return []

        if (self._is_training):
            return self._global_step.value
        else:
            update_op = xdl.ps_assign_add_op(
                var_name = self._global_step.name,
                var_type = self._global_step.vtype,
                delta = np.array(1, dtype=np.int64))
            return [self._global_step.value, update_op]

    def after_run(self, v):
        if self._ckpt_dir is None:
            return
        global_step = v[0] if isinstance(v, list) else v
        if self._first_run:
            self._save_cnt = global_step / self._save_interval
            self._first_run = False
        if global_step > 0 and global_step / self._save_interval > self._save_cnt:
            version = self._create_version(global_step)
            print('save checkpoint at global_step[%d], ckpt version[%s]' % 
                  (global_step, version))
            self._saver.save(version)
            self._save_cnt = self._save_cnt + 1
            if self._export_graph:
                self._saver.export_graph(as_text=self._as_text);

    def _create_version(self, global_step):
        return "ckpt-{:.>20}".format(global_step)
