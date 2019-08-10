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

import xdl

import os
import numpy as np
import json
import time
import datetime
from xdl.python.lib.graph import current_graph
from xdl.python.proto import graph_def_pb2
from google.protobuf.text_format import Parse as TextParse
from xdl.python.utils.file_io import write_string_to_file
from xdl.python.framework.session import Hook
from xdl.python.training.export import output
from xdl.python.training.training_utils import get_global_step
from xdl.python.lib.graph import execute
from xdl.python.utils.config import get_ckpt_dir
from xdl.python.lib.tensor import Tensor
from xdl.python.utils.file_io import write_string_to_file

def _string_to_int8(src):
    return np.array([ord(ch) for ch in src], dtype=np.int8)

def _graphdef_to_pb(graph_def):
    pb = graph_def_pb2.GraphDef()
    TextParse(graph_def.to_proto_string(), pb)
    return pb

'''
Usage:
    xdl.graph_tag().set_input(data_io)
    xdl.graph_tag().set_mx_output(prop)    # in xdl.mxnet_wrapper
    xdl.graph_tag().set_tf_output(prop)    # in xdl.tf_wrapper
'''
class GraphTag(object):
    def __init__(self):
        self._inputs = list()
        self._output_op_name = 'default'
        self._sparse_list = list()
        self._fea_dict = dict()
    def append_input(self, op_name, input_name, type, size=1, table=0):
        if type == xdl.features.sparse:
            input_type = graph_def_pb2.kSparse
        else:
            input_type = graph_def_pb2.kDense
        self._inputs.append((op_name, input_name, input_type, size, table))
    def set_input(self, data_io):
        for (idx, name, type, nvec, table) in data_io.tags:
            if type == xdl.features.sparse:
                input_type = graph_def_pb2.kSparse
            else:
                input_type = graph_def_pb2.kDense
            self._inputs.append(('/GetBatch:%d' % idx, name, input_type, nvec, table))
        self._sparse_list = data_io._sparse_list
        self._fea_dict = data_io._fea_dict
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
    def output_op_name(self):
        return self._output_op_name
    @property
    def sparse_list(self):
        return self._sparse_list
    @property
    def fea_dict(self):
        return self._fea_dict

_GRAPH_TAG = GraphTag()

def graph_tag():
    global _GRAPH_TAG
    return _GRAPH_TAG

class EmbedOpArg(object):
    def __init__(self, str, dig, desc):
        self.str = str
        self.dig = dig
        self.desc = desc

class EmbedCodeConf(object):
    def __init__(self, name, fea_type, emb_dim, fea_groupids, op, op_args=None):
        self.name = name
        self.fea_type = fea_type
        self.emb_dim = emb_dim
        self.fea_groupids = fea_groupids
        self.op = op
        self.op_args = op_args

class Saver(object):
    def __init__(self, ckpt_dir=None, tf_graph_name=None):
        self._ckpt_dir = ckpt_dir
        if self._ckpt_dir is None:
          self._ckpt_dir = get_ckpt_dir()
        self._graph_def = _graphdef_to_pb(current_graph()._graph_def)
        self._tf_graph_name = tf_graph_name
    def save(self, version):
        execute(self.save_op(version))
    
    def save_meta(self, version, **kwargs):
      kwargs['xdl_global_step'] = get_global_step().value
      values = []
      for k, v in kwargs.iteritems():
        if isinstance(v, Tensor):
          v = execute(v)
          values.append(v)
        else:
          values.append(np.array(v))
      values = [min(v.flatten().tolist()) for v in values]
      keys = kwargs.keys()
      assert len(keys) == len(values)
      buf = json.dumps(dict(zip(keys, values)))
      path = os.path.join(self._ckpt_dir, version, '.meta')
      write_string_to_file(path, buf)
    def restore(self, version):
        execute(self.restore_op(version))
    def save_op(self, version):
        return xdl.ps_save_op(_string_to_int8(version))
    def restore_op(self, version):
        return xdl.ps_restore_op(_string_to_int8(version))
    def export_graph(self, output_dir, as_text=False):
        print 'start export graph'
        for (op_name, input_name, input_type, size, table) in graph_tag().inputs:
            self.append_input(op_name, input_name, input_type, size, table)
        self._graph_def.tag.output.op_name = graph_tag().output_op_name
        if as_text:
            write_string_to_file("graph.txt", str(self._graph_def))
            output(output_dir, "graph.txt")
        else:
            write_string_to_file("graph.pb", self._graph_def.SerializeToString())
            output(output_dir, "graph.pb")
        print 'finish export graph'
    def append_input(self, op_name, input_name, input_type, size=1, table=0):
        inp = self._graph_def.tag.input.add()
        inp.op_name = op_name
        inp.input_name = input_name
        inp.input_type = input_type
        inp.size = size
        inp.table = table
    def export_sparse_conf_v4(self, sparse_v4_dir, space_list=list(list())):
        print 'start export sparse conf v4'
        from xdl.python.training.v4 import f2id_pb2
        from google.protobuf.text_format import MessageToString
        f2id_list = f2id_pb2.F2IdList()
        for gid in xrange(len(space_list)):
            for feature_groupid in space_list[gid]:
                f2_id = f2id_list.item.add()
                f2_id.feature_groupid = feature_groupid
                f2_id.fid = gid
        with open('embed.best.meta', 'wb') as f:  # f2id.pb
            f.write(MessageToString(f2id_list))
        output(sparse_v4_dir, 'embed.best.meta')
        print 'finish export sparse conf v4'
    def export_dense_conf_v4(self, dense_v4_dir, embed_code_conf_list):
        print 'start export dense conf v4'
        from xdl.python.training.v4 import dense_input_conf_pb2, embed_dimension_xdl_code_pb2
        from google.protobuf.text_format import MessageToString
        embed_dim_list = embed_dimension_xdl_code_pb2.EmbedDimList()
        embed_dim_list.model_signature = str(long(time.mktime(datetime.datetime.utcnow().timetuple()) * 1000000))
        global_offset = 1
        index = 0
        block = None
        for embed_conf in embed_code_conf_list:
            if block is None or embed_conf.name != block.name:
                block = embed_dim_list.dense_input.input_blocks.add()
                block.name = embed_conf.name
                if embed_conf.fea_type == 'common':
                    block.hint_type = dense_input_conf_pb2.COMMON
                elif embed_conf.fea_type == 'ncommon':
                    block.hint_type = dense_input_conf_pb2.UNCOMMON
                else:
                    block.hint_type = dense_input_conf_pb2.UNKNOWN
                index = 0
            for fea_groupid in embed_conf.fea_groupids:
                embed_dim = embed_dim_list.embed_dim_list.add()
                embed_dim.embed_dim = embed_conf.emb_dim
                embed_dim.fea_groupid = fea_groupid
                embed_dim.fea_type = 'common' if embed_conf.fea_type == 'unknown' else embed_conf.fea_type  # 'common' | 'ncommon'
                embed_dim.fea_group_global_offset = global_offset
                global_offset += 1
                embed_field = block.embed_fields.add()
                embed_field.index = index
                index += 1
                embed_field.dim = embed_conf.emb_dim
                embed_field.fea_group_id = fea_groupid
                embed_field.op = dense_input_conf_pb2.KSUM if embed_conf.op == 'ksum' else dense_input_conf_pb2.ASSIGN
                if embed_conf.op_args is not None:
                    for arg in embed_conf.op_args:
                        op_arg = embed_field.op_args.add()
                        if arg.str is not None:
                           op_arg.str = arg.str
                        if arg.dig is not None:
                           op_arg.dig = arg.dig
                        if arg.desc is not None:
                           op_arg.desc = arg.desc
        with open('embed-dim-xdl-code-conf', 'wb') as f:  # dense_input_conf.pb
            f.write(MessageToString(embed_dim_list))
        output(dense_v4_dir, 'embed-dim-xdl-code-conf')
        output(dense_v4_dir, 'network_desc.pb')
        print 'finish export dense conf v4'
    def export_graph_v4(self, output_v4_dir, emb_dim, space_list=list(list()), order=1, comm_hint_type='common'):
        print 'start export graph v4 @deprecated'
        self.export_sparse_graph_v4(output_v4_dir, emb_dim, space_list, order, comm_hint_type)
        print 'finish export graph v4 @deprecated'
    def export_sparse_graph_v4(self, output_v4_dir, emb_dim, space_list, order=1, comm_hint_type='common'):
        from xdl.python.training.v4 import dense_input_conf_pb2, embed_dimension_xdl_code_pb2, f2id_pb2
        from google.protobuf.text_format import MessageToString
        embed_dim_list = embed_dimension_xdl_code_pb2.EmbedDimList()
        embed_dim_list.model_signature = str(long(time.mktime(datetime.datetime.utcnow().timetuple()) * 1000000))
        if order == 1:
            comm_block = embed_dim_list.dense_input.input_blocks.add()
            ncomm_block = embed_dim_list.dense_input.input_blocks.add()
        else:
            ncomm_block = embed_dim_list.dense_input.input_blocks.add()
            comm_block = embed_dim_list.dense_input.input_blocks.add()
        comm_block.name = 'comm'
        ncomm_block.name = 'ncomm'
        comm_block.hint_type = dense_input_conf_pb2.UNKNOWN if comm_hint_type == 'unknown' else dense_input_conf_pb2.COMMON
        ncomm_block.hint_type = dense_input_conf_pb2.UNCOMMON
        comm_index = 0
        ncomm_index = 0
        for i in xrange(len(graph_tag().sparse_list)):
            name = graph_tag().sparse_list[i]
            table = graph_tag().fea_dict[name]['table']
            embed_dim = embed_dim_list.embed_dim_list.add()
            embed_dim.embed_dim = emb_dim
            embed_dim.fea_groupid = name
            embed_dim.fea_group_global_offset = i + 1
            embed_dim.fea_type = "ncommon" if table == 0 else "common"
            embed_field = ncomm_block.embed_fields.add() if table == 0 else comm_block.embed_fields.add()
            if table == 0:
                embed_field.index = ncomm_index
                ncomm_index += 1
            else:
                embed_field.index = comm_index
                comm_index += 1
            embed_field.dim = emb_dim
            embed_field.fea_group_id = name
            embed_field.op = dense_input_conf_pb2.KSUM
        with open('embed-dim-xdl-code-conf', 'wb') as f:  # dense_input_conf.pb
            f.write(MessageToString(embed_dim_list))
        output(output_v4_dir, 'embed-dim-xdl-code-conf')
        f2id_list = f2id_pb2.F2IdList()
        for gid in xrange(len(space_list)):
            for feature_groupid in space_list[gid]:
                f2_id = f2id_list.item.add()
                f2_id.feature_groupid = feature_groupid
                f2_id.fid = gid
        with open('embed.best.meta', 'wb') as f:  # f2id.pb
            f.write(MessageToString(f2id_list))
        output(output_v4_dir, 'embed.best.meta')

class CheckpointMeta:
  def __init__(self, **kwargs):
    self._meta = {}
    for k, v in kwargs.iteritems():
      if not isinstance(v, Tensor):
        continue
      self._meta[k] = v
    self._meta['xdl_global_step'] = get_global_step().value

  def save(self, ckpt_dir, version, values):
    if len(self._meta) == 0:
      return
    keys = self._meta.keys()
    values = [min(v.flatten().tolist()) for v in values]

    assert len(keys) == len(values)
    buf = json.dumps(dict(zip(keys, values)))
    path = os.path.join(ckpt_dir, version, '.meta')
    write_string_to_file(path, buf)

  def values(self):
    return self._meta.values()

class CheckpointHook(Hook):
    def __init__(self, save_interval_step=None, save_interval_secs=None,
                 is_training=True, meta=None, tf_backend=False, max_to_keep=5,
                 tf_graph_name=None):
        super(CheckpointHook, self).__init__(priority=3000)
        self._global_step = get_global_step()
        self._save_interval_step = save_interval_step
        self._save_interval_secs = save_interval_secs
        self._ckpt_dir = get_ckpt_dir()
        self._saver = Saver(self._ckpt_dir, tf_graph_name)
        self._is_training = is_training
        self._last_save_step = 0
        self._last_save_time = time.time()
        self._meta = meta
        self._max_to_keep = max_to_keep
        self._ckpt_queue = []

        if self._save_interval_step is None and self._save_interval_secs is None:
            print("Checkpoint interval_steps and interval_secs both not set, use default 10000 steps.")
            self._save_interval_step = 10000
        elif self._save_interval_step is not None and self._save_interval_secs is not None:
            raise ValueError("Checkpoint interval_steps and interval_secs can't be both set.")

        self.gstep_val = 0
        self.meta_val = None

    def before_run(self, v):
        if self._ckpt_dir is None:
            return []

        if (self._is_training):
            res = [self._global_step.value]
            if self._meta is not None:
              res.extend(self._meta.values())
            return res

    def after_run(self, v):
        if self._ckpt_dir is None:
            return
        self.gstep_val = v[0] if isinstance(v, list) else v
        if self._meta:
            self.meta_val = v[1:]
        if self._save_interval_step is not None:
            if self.gstep_val - self._last_save_step >= self._save_interval_step:
                self._save_ckpt(self.gstep_val, self.meta_val)
                self._last_save_step = self.gstep_val
                self._ckpt_queue.append(self._create_version(self.gstep_val))
                self._check_ckpt_queue()
        elif self._save_interval_secs is not None:
            if time.time() - self._last_save_time >= self._save_interval_secs:
                self._save_ckpt(self.gstep_val, self.meta_val)
                self._last_save_time = time.time()
                self._ckpt_queue.append(self._create_version(self.gstep_val))
                self._check_ckpt_queue()

    def end(self):
        self.gstep_val = xdl.execute(self._global_step.value) + 1
        if self._meta is not None:
            self.meta_val = xdl.execute(self._meta.values())
        self._save_ckpt(self.gstep_val, self.meta_val)
        if self.gstep_val != self._last_save_step:
            self._ckpt_queue.append(self._create_version(self.gstep_val))
        self._check_ckpt_queue()

    def _check_ckpt_queue(self):
        while len(self._ckpt_queue) > self._max_to_keep:
            del_version = self._ckpt_queue.pop(0)
            del_ckpt = self._ckpt_dir.rstrip("/") + "/" + del_version
            if self._ckpt_dir.startswith('hdfs://'):
                cmd = "hadoop fs -rm -r %s" % del_ckpt
            else:
                cmd = "rm -rf %s" % del_ckpt
            ret = os.system(cmd)
            if ret == 0:
                print("ckpt number is larger than max_to_keep setting, delete oldest ckpt %s" % del_ckpt)
            else:
                raise ValueError("Failed: %s" % cmd)

    def _save_ckpt(self, global_step, meta_values):
        version = self._create_version(global_step)
        print('save checkpoint at global_step[%d], ckpt version[%s]' % (global_step, version))
        self._saver.save(version)
        if meta_values:
            self._meta.save(self._ckpt_dir, version, meta_values)

    def _create_version(self, global_step):
        return "ckpt-{:.>20}".format(global_step)

class RestoreFromHook(Hook):
    def __init__(self, ckp_model):
        super(RestoreFromHook, self).__init__(priority=1000)
        self._ckp_model = ckp_model
        self._saver = Saver(get_ckpt_dir())
    def create_session(self):
        if self._ckp_model and len(self._ckp_model) > 0:
            if xdl.get_task_index() == 0:
                self._saver.restore(self._ckp_model)
                print("restore checkpoint from " + str(self._ckp_model))
            else:
                time.sleep(120)
