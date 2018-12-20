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

#!/usr/bin/env python

from __future__ import print_function

import sys
import os

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CUR_DIR, '..'))

import struct
import numpy as np

from store import KVItem
import tree_pb2 as tree_proto


class TreeBuilder:
    def __init__(self, filename):
        self.filename = filename

    def build(self, ids, codes, data=None,
              id_offset=None, stat=None, kv_file=None):
        # process id offset
        if not id_offset:
            max_id = 0
            for id in ids:
                if id > max_id:
                    max_id = id
            id_offset = max_id + 1

        # sort by codes
        argindex = np.argsort(codes)
        codes = codes[argindex]
        ids = ids[argindex]
        data = data[argindex]

        # Trick, make all leaf nodes to be in same level
        min_code = 0
        max_code = codes[-1]
        while max_code > 0:
            min_code = min_code * 2 + 1
            max_code = int((max_code - 1) / 2)

        for i in range(len(codes)):
            while codes[i] < min_code:
                codes[i] = codes[i] * 2 + 1

        if kv_file:
            with open(kv_file, 'w') as f:
                for id, code, datum in zip(ids, codes, data):
                    f.write('{}, {}'.format(id, self._make_prefix_code(code)))
                    if isinstance(datum, list):
                        for d in datum:
                            f.write(', {}'.format(d))
                    f.write('\n')

        pstat = None
        if stat:
            pstat = dict()
            for id, code in zip(ids, codes):
                ancs = self._ancessors(code)
                for anc in ancs:
                    if id in stat:
                      if anc not in pstat:
                        pstat[anc] = 0.0
                      pstat[anc] += stat[id]

        seq = 1
        filter_set = set()
        meta = tree_proto.TreeMeta()
        meta.max_level = 0
        id_code_part = []
        with open(self.filename, 'wb') as f:
            for id, code, datum in zip(ids, codes, data):
                node = tree_proto.Node()
                node.id = id
                node.is_leaf = True
                node.probality = stat[id] if stat and id in stat else 1.0
                node.leaf_cate_id = 0
                if isinstance(datum, list):
                    for d in datum:
                        node.embed_vec.append(d)
                kv_item = KVItem()
                kv_item.key = self._make_key(code)
                kv_item.value = node.SerializeToString()
                self._write_kv(f, kv_item.SerializeToString())
                ancessors = self._ancessors(code)
                if len(ancessors) + 1 > meta.max_level:
                    meta.max_level = len(ancessors) + 1
                if not id_code_part or \
                   len(id_code_part[-1].id_code_list) == 512:
                    part = tree_proto.IdCodePart()
                    part.part_id = 'Part_' + \
                        self._make_key(len(id_code_part) + 1)
                    id_code_part.append(part)
                part = id_code_part[-1]
                id_code = part.id_code_list.add()
                id_code.id = id
                id_code.code = code
                for ancessor in ancessors:
                    if ancessor not in filter_set:
                        node = tree_proto.Node()
                        node.id = id_offset + ancessor  # id = id_offset + code
                        node.is_leaf = False
                        node.leaf_cate_id = 0
                        node.probality = pstat[ancessor] \
                            if pstat and ancessor in pstat else 1.0
                        kv_item = KVItem()
                        kv_item.key = self._make_key(ancessor)
                        kv_item.value = node.SerializeToString()
                        self._write_kv(f, kv_item.SerializeToString())
                        seq += 1
                        filter_set.add(ancessor)

            for part in id_code_part:
                meta.id_code_part.append(part.part_id)
                kv_item = KVItem()
                kv_item.key = part.part_id
                kv_item.value = part.SerializeToString()
                self._write_kv(f, kv_item.SerializeToString())

            kv_item = KVItem()
            kv_item.key = '.tree_meta'
            kv_item.value = meta.SerializeToString()
            self._write_kv(f, kv_item.SerializeToString())

    def _ancessors(self, code):
        ancs = []
        while code > 0:
            code = int((code - 1) / 2)
            ancs.append(code)
        return ancs

    def _make_key(self, code):
        return struct.pack('L', code)[::-1]

    def _write_kv(self, fwr, message):
        fwr.write(struct.pack('i', len(message)))
        fwr.write(message)

    def _make_prefix_code(self, code):
        prefix = ''
        while code > 0:
            if code % 2 == 0:
                prefix += '1'
            else:
                prefix += '0'
            code = int((code - 1) / 2)
        prefix += '0'
        return prefix[::-1]
