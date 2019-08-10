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
from xdl.python import pybind
from xdl.python.utils.collections import READER_HOOKS, add_to_collection
from xdl.python.framework.variable import Variable
from xdl.python.lib.graph import execute
import numpy as np
import copy

class DataIO(pybind.DataIO):
    def __init__(self, ds_name, file_type=pybind.parsers.txt,
                 fs_type = pybind.fs.local, namenode="", enable_state=True,
                 save_state_interval=100, global_schedule=False):
        self._ds_name = ds_name
        self._fs_type = fs_type
        super(DataIO, self).__init__(name=ds_name, file_type=file_type,
                                     fs_type=fs_type, namenode=namenode,
                                     worker_id=xdl.get_task_index(),
                                     global_schedule=global_schedule)
        self._sparse_list = list()
        self._dense_list = list()
        self._fea_dict = dict()
        self._nindicator = 0
        self._batch_size = 0
        self._label_count = 1
        self._unique_ids = False
        self._keep_skey = False

        self._init_tags()
        self._hook = None

        if isinstance(enable_state, DataIO):
            self.failover_with(enable_state)
        elif enable_state:
            self._init_reader_state_hook(global_schedule, save_state_interval)

    def _init_reader_state_hook(self, global_schedule, save_state_interval):
        if global_schedule:
            from xdl.python.io.global_reader_state_hook import GlobalReaderStateHook
            self._hook = GlobalReaderStateHook(self)
            add_to_collection(READER_HOOKS, self._hook)
        else:
            from xdl.python.io.reader_state_hook import ReaderStateHook
            self._hook = ReaderStateHook(self, save_state_interval)
            add_to_collection(READER_HOOKS, self._hook)

    def failover_with(self, io):
        assert isinstance(io, DataIO)
        hook = copy.copy(io._hook)
        add_to_collection(READER_HOOKS, hook)

    def add_path(self, path):
        '''
            add single string or list of strings as path,
            it's a low level interface,
            non-stream system should use DataSharding to partition files
        '''
        if isinstance(path, basestring):
            super(DataIO, self).add_path(path)
            return
        for p in path:
            super(DataIO, self).add_path(p)
        return self

    def feature(self, name, type, table=0, nvec=0, mask="", serialized=False, cutoff=0, dsl=""):
        assert type in (pybind.features.sparse, pybind.features.dense)
        if type == pybind.features.sparse:
            self._sparse_list.append(name)
        else:
            assert nvec > 0
            self._dense_list.append(name)

        if len(mask) != 0:
            assert len(mask) == nvec, "mask length(%d) must equal to nvec(%d)"%(nvec, len(mask))
            mask = mask[::-1] # as std::bitset init from end of string

        self._fea_dict[name] = {'type':type, 'table':table, 'nvec':nvec, 'mask':mask,
                                'serialized':serialized, 'cutoff': cutoff}

        if self._nindicator < table:
            self._nindicator = table

        super(DataIO, self).feature(name, type, table, nvec, mask, serialized, cutoff, dsl)
        return self

    def epochs(self, epochs):
        self._epochs = epochs;
        super(DataIO, self).epochs(epochs);
        return self

    def threads(self, threads):
        super(DataIO, self).threads(threads);
        return self

    def batch_size(self, batch_size):
        self._batch_size = batch_size
        super(DataIO, self).batch_size(batch_size)
        return self

    def label_count(self, label_count):
        self._label_count = label_count
        super(DataIO, self).label_count(label_count)
        return self

    def split_group(self, split_group):
        self._split_group = split_group
        super(DataIO, self).split_group(split_group)
        return self

    def keep_skey(self, keep_skey):
        self._keep_skey = keep_skey
        super(DataIO, self).keep_skey(keep_skey)
        return self

    def unique_ids(self, unique_ids):
        self._unique_ids = unique_ids
        super(DataIO, self).unique_ids(unique_ids)
        return self

    def load_feature(self, feature_description):
        dict = {}
        for kv in feature_description.split(';'):
            kvs = kv.partition('=')
            assert len(kvs) == 3
            assert kvs[1] == '='
            dict[kvs[0].strip()] = kvs[2].strip()

        assert dict.has_key('Name')
        if dict.has_key('Type') and dict['Type'] == 'Vector':
            type = pybind.features.dense
        else:
            type = pybind.features.sparse

        if dict.has_key('Table'):
            table = int(dict['Table'])
        else:
            table = 0

        if dict.has_key('Nvec'):
            nvec = int(dict['Nvec'])
        else:
            nvec = 0

        if dict.has_key('Dim'):
            dim = int(dict['Dim'])
        else:
            dim = 0

        if dict.has_key('Serialized'):
            serialized = True
        else:
            serialized = False

        has_key = dict.has_key('Expr')
        dsl = ''
        if has_key:
            if len(dict['Expr'].partition('(')) < 3:
                has_key = False
            else:
                dsl = feature_description
        self.feature(name=dict['Name'], type=type, table=table, nvec=nvec, serialized=serialized, dsl=dsl)
        return has_key

    def load_feature_conf(self, feature_conf):
        has_expr = False
        f = open(feature_conf, 'r')
        for line in f:
            line = line.strip(' \n')
            if len(line) != 0:
                has_expr |= self.load_feature(line)

        f.close()
        if has_expr:
            self._feature_op = pybind.GetIOP("FeatureOP")
            self.add_op(self._feature_op)

    @property
    def ds_name(self):
        return self._ds_name

    @property
    def fs_type(self):
        return self._fs_type

    @property
    def cache_count(self):
        return self._cache_count

    def set_prop(self, prop):
        return xdl.set_prop(ds=self._ds_name, prop=prop)

    def check_finish(self):
        return xdl.check_finish(ds=self._ds_name)

    def release_batch(self):
        return xdl.release_batch(ds=self._ds_name)

    @property
    def tags(self):
        '''
            list((idx, name, type, nvec, table))
        '''
        return self._tags

    def _init_tags(self):
        self._tags = list()
        self._input_idx = 0

    def _append_tag(self, name, type, nvec=1, table=0):
        self._tags.append((self._input_idx, name, type, nvec, table))
        self._input_idx = self._input_idx + 1

    def append_tags(self):
        for i in range(self._nindicator):
            self._append_tag("indicator.%d" % i, pybind.features.dense)

        for name in self._sparse_list:
            fea_dict = self._fea_dict[name]
            self._append_tag("%s.indices" % name, fea_dict['type'], fea_dict['nvec'], fea_dict['table'])
        for name in self._sparse_list:
            fea_dict = self._fea_dict[name]
            self._append_tag("%s.ids" % name, fea_dict['type'], fea_dict['nvec'], fea_dict['table'])
        for name in self._sparse_list:
            fea_dict = self._fea_dict[name]
            self._append_tag("%s.segments" % name, fea_dict['type'], fea_dict['nvec'], fea_dict['table'])
        for name in self._sparse_list:
            fea_dict = self._fea_dict[name]
            self._append_tag("%s.values" % name, fea_dict['type'], fea_dict['nvec'], fea_dict['table'])

        for name in self._dense_list:
            fea_dict = self._fea_dict[name]
            self._append_tag("%s.values" % name, fea_dict['type'], fea_dict['nvec'], fea_dict['table'])

        self._append_tag("skbuf", pybind.features.dense)
        self._append_tag("label", pybind.features.dense)

    def feature_option(self, feature_name):
        if feature_name not in self._fea_dict:
            return None
        return self._fea_dict[feature_name]

    def get_reader_offset(self):
      assert self._hook is not None, "get reader offset needs enable state"
      return self._hook.get_reader_offset()

    def read(self):
        assert self._batch_size > 0
        assert self._label_count > 0

        out = xdl.get_batch(ds=self._ds_name,
                            sparse_count=len(self._sparse_list),
                            dense_count=len(self._dense_list),
                            indicator_count=self._nindicator,
                            tag_cnt=None,
                            dtype=xdl.DataType.float)
        batch = dict()
        batch["indicators"] = out[0]
        batch["_indices"] = out[1]
        batch["_ids"] = out[2]
        batch["_segments"] = out[3]
        batch["_svalues"] = out[4]
        batch["_dvalues"] = out[5]
        batch["_sidx"] = out[6]
        batch["_sseg"] = out[7]
        if self._keep_skey:
            batch["skbuf"] = out[8]
        batch["label"] = out[9]
        batch["_tag_"] = out[10]

        ### indicator
        for i in range(len(batch["indicators"])):
            batch["indicators"][i].set_shape([self._batch_size])

        ### sparse
        assert len(self._sparse_list) == len(batch['_indices'])
        assert len(self._sparse_list) == len(batch['_ids'])
        assert len(self._sparse_list) == len(batch['_svalues'])
        assert len(self._sparse_list) == len(batch['_segments'])
        assert len(self._sparse_list) == len(batch['_sidx'])
        assert len(self._sparse_list) == len(batch['_sseg'])

        for i in range(len(self._sparse_list)):
            name = self._sparse_list[i]
            batch[name] = xdl.SparseTensor(batch['_ids'][i], batch["_svalues"][i], batch['_segments'][i],
                                           batch['_indices'][i] if self._unique_ids else None,
                                           batch['_sidx'][i] if self._unique_ids else None,
                                           batch['_sseg'][i] if self._unique_ids else None)
            opt = self._fea_dict.get(name)
            assert opt != None
            assert opt['type'] == pybind.features.sparse

            batch[name].set_shape([self._batch_size])
            batch[name].set_name(name)
            batch['_ids'][i].set_name(name)
            batch['_indices'][i].set_name(name)
            batch['_sidx'][i].set_name(name)
            batch['_sseg'][i].set_name(name)

        ### dense
        assert len(self._dense_list) == len(batch['_dvalues'])
        for i in range(len(self._dense_list)):
            name = self._dense_list[i]
            batch[name] = batch["_dvalues"][i]

            opt = self._fea_dict.get(name)
            assert opt != None
            assert opt['type'] == pybind.features.dense
            assert opt['nvec'] > 0

            batch[name].set_shape((self._batch_size, opt['nvec']))
            batch[name].set_name(name)

        ### label
        batch['label'].set_shape([self._batch_size, self._label_count])
        batch['label'].set_name('label')

        ### sample_tag
        batch["_tag_"].set_shape([])

        ### tags
        self.append_tags()

        return batch
