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
from xdl.python.ops.py_func import dtype_xdl_2_np
from xdl.python.sparse_engine.base import MergedSparseTensor

""" python wrapper for op used in sparse_engine."""

def ksum(embeddings, idx, values, segments, sidx, sseg, device='CPU', **device_attrs):
    groups = np.array([], dtype=dtype_xdl_2_np(segments.dtype))
    with xdl.device(device, **device_attrs):
        res = xdl.ksum(embeddings, idx, values, segments, groups, sidx, sseg)
    return res

def kmean(embeddings, idx, values, segments, sidx, sseg, device='CPU', **device_attrs):
    groups = np.array([], dtype=dtype_xdl_2_np(segments.dtype))
    with xdl.device(device, **device_attrs):
        res = xdl.ksum(embeddings, idx, values, segments, groups, sidx, sseg, average=True)
    return res

def merged_ksum(embeddings, idx, values, segments,
                groups, sidx, sseg, device='CPU', **device_attrs):
    with xdl.device(device, **device_attrs):
        res = xdl.ksum(embeddings, idx, values, segments, groups, sidx, sseg)
    return res

def merged_kmean(embeddings, idx, values, segments,
                 groups, sidx, sseg, device='CPU', **device_attrs):
    with xdl.device(device, **device_attrs):
        res = xdl.ksum(embeddings, idx, values, segments, groups, sidx, sseg, average=True)
    return res

def tile(embeddings, idx, values, segments, length,
         reverse=False, device='CPU', **device_attrs):
    groups = np.array([], dtype=dtype_xdl_2_np(segments.dtype))
    with xdl.device(device, **device_attrs):
        res = xdl.tile(embeddings, idx, values, segments, groups,
                        reverse=reverse, length=length)
    return res

def merged_tile(embeddings, idx, values, segments, groups, length,
                reverse=False, device='CPU', **device_attrs):
    with xdl.device(device, **device_attrs):
        res = xdl.tile(embeddings, idx, values, segments, groups,
                        reverse=reverse, length=length)
    return res

def merge_sparse(sparse_inputs, device='CPU', **device_attrs):
    id_list = [x.ids for x in sparse_inputs]
    value_list = [x.values for x in sparse_inputs]
    segment_list = [x.segments for x in sparse_inputs]
    with xdl.device(device, **device_attrs):
        ids, values, segments, groups = \
            xdl.merge_sparse_op(id_list, value_list, segment_list)
    return MergedSparseTensor(ids, values, segments, groups)

def take(feature, indicator, device='CPU', **device_attrs):
    with xdl.device(device, **device_attrs):
        res = xdl.take_op(feature, indicator)
    return res
