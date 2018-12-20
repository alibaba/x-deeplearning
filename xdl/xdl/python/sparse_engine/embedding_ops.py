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

import xdl
import numpy as np
from xdl.python.ops.py_func import dtype_xdl_2_np

""" python wrapper for op used in sparse_engine."""

def ksum(embeddings, idx, values, segments):
    groups = np.array([], dtype=dtype_xdl_2_np(segments.dtype))
    return xdl.ksum(embeddings, idx, values, segments, groups)

def kmean(embeddings, idx, values, segments):
    groups = np.array([], dtype=dtype_xdl_2_np(segments.dtype))
    return xdl.ksum(embeddings, idx, values, segments, groups, average=True)

def merged_ksum(embeddings, idx, values, segments, groups):
    return xdl.ksum(embeddings, idx, values, segments, groups)

def merged_kmean(embeddings, idx, values, segments, groups):
    return xdl.ksum(embeddings, idx, values, segments, groups, average=True)

def tile(embeddings, idx, values, segments, length, reverse=False):
    groups = np.array([], dtype=dtype_xdl_2_np(segments.dtype))
    return xdl.tile(embeddings, idx, values, segments, groups,
                    reverse=reverse, length=length)

def merged_tile(embeddings, idx, values, segments, groups, length, reverse=False):
    return xdl.tile(embeddings, idx, values, segments, groups,
                    reverse=reverse, length=length)


def merge_sparse(sparse_inputs):
    id_list = [x.ids for x in sparse_inputs]
    value_list = [x.values for x in sparse_inputs]
    segment_list = [x.segments for x in sparse_inputs]
    ids, values, segments, groups = \
        xdl.merge_sparse_op(id_list, value_list, segment_list)
    return MergedSparseTensor(ids, values, segments, groups)

def take(feature, indicator):
    return xdl.take_op(feature, indicator)
