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
from xdl.python.lib.datatype import *
from xdl.python.framework.variable import VarType
from xdl.python.sparse_engine.embedding_ops import merge_sparse

_EMBEDDING_INFO = {}
_EMBEDDING_TENSOR = {}

class EmbeddingInfo(object):
    def __init__(self, name, feature_dim, emb_dim, combiner, output_tensor, xdl_var, tile_length, embedding):
        """used to save some embedding information needed by ps."""
        self._name = name
        self._feature_dim = feature_dim
        self._emb_dim = emb_dim
        self._combiner = combiner
        self._output_tensor = output_tensor
        self._var = xdl_var
        self._tile_length = tile_length
        self._embedding = embedding
    @property
    def name(self):
        return self._name
    @property
    def feature_dim(self):
        return self._feature_dim
    @property
    def emb_dim(self):
        return self._emb_dim
    @property
    def combiner(self):
        return self._combiner
    @property
    def output_tensor(self):
        return self._output_tensor
    @property
    def var(self):
        return self._var
    @property
    def tile_length(self):
        return self._tile_length
    @property
    def embedding(self):
        return self._embedding

def get_embedding_info(key):
    var = induce_emb_var_from_tensor(key)
    if var is not None:
        return get_embedding_info_by_var(var)
    return None

def get_embedding_info_by_var(key):
    global _EMBEDDING_INFO
    if key in _EMBEDDING_INFO:
        return _EMBEDDING_INFO[key]
    else:
        return None

def get_embedding_output(key):
    info = get_embedding_info_by_var(key)
    if info is None:
        return None
    return info.output_tensor

def get_embedding_info_by_name(name):
    global _EMBEDDING_INFO
    for var in _EMBEDDING_INFO:
        if var.name == name:
            return _EMBEDDING_INFO[var]
    return None

def set_embedding_info(inputs, name, feature_dim, emb_dim, combiner):
    global _EMBEDDING_INFO
    for item in inputs:
        if item in _EMBEDDING_INFO:
            raise Exception('dupcate key in embedding info dict:' + str(item))
        _EMBEDDING_INFO[item] = EmbeddingInfo(name, feature_dim, emb_dim, combiner)

def set_embedding_info(inputs, emb_info):
    global _EMBEDDING_INFO
    for item in inputs:
        if item in _EMBEDDING_INFO:
            raise Exception('dupcate key in embedding info dict:' + str(item))
        _EMBEDDING_INFO[item] = emb_info

def is_embedding_var(var):
    global _EMBEDDING_INFO
    return var in _EMBEDDING_INFO

def induce_emb_var_from_tensor(t):
    global _EMBEDDING_TENSOR
    q = []
    q.extend(t.op.inputs)
    while len(q) > 0:
        i = q.pop(0)
        q.extend(i.op.inputs)
        if i in _EMBEDDING_TENSOR:
            return _EMBEDDING_TENSOR[i]
    return None    

EMBEDDING_LIST = list()
EMBEDDING_SET = set()

def embedding_list():
    global EMBEDDING_LIST
    return EMBEDDING_LIST

def embedding(name, sparse_input, initializer, emb_dim, feature_dim,
              combiner='sum',
              vtype=VarType.Index,
              length=50,
              reverse=False,
              batch_read=3000,
              feature_add_probability=1.0,
              cbf=0,
              device='CPU',
              statis_list=None,
              statis_decay=0.07,
              statis_decay_period=100,
              labels=None,
              save=True,
              **device_attr):
    """xdl embedding
       Args:
         name: name for embedding, will be used for declaring variable on ps-plus
         sparse_input: a sparse tensor represent input data
         initializer: intializer for the variable on ps-plus
         emb_dim: embedding dimension
         feature_dim: sparse input dimension, for pre-allocate memory
         combiner: reduce operator, support sum|mean
       Returns:
         a tensor represent embedding result
       Raises:
         None
    """

    global EMBEDDING_LIST, EMBEDDING_SET
    if name not in EMBEDDING_SET:
        EMBEDDING_SET.add(name)
        EMBEDDING_LIST.append(name)

    import xdl.python.framework.variable as variable
    with variable.variable_info(batch_read=batch_read, save_ratio=feature_add_probability, bloom_filter=cbf, save="true" if save else "false"):
        var = variable.Variable(name=name,
                                dtype=DataType.float,
                                shape=[feature_dim, emb_dim],
                                initializer=initializer,
                                vtype=vtype,
                                trainable=True)
        if statis_list is not None:
            statis_vars = []
            for statis_type in statis_list:
                statis_var = variable.Variable(name=name,
                                               dtype=DataType.float,
                                               shape=[feature_dim, 1],
                                               initializer=xdl.Zeros(),
                                               vtype=vtype,
                                               trainable=False,
                                               statis_type=statis_type,
                                               statis_decay=statis_decay,
                                               statis_decay_period=statis_decay_period)
                statis_vars.append(statis_var)

    if sparse_input.has_unique_ids():
        unique_ids = xdl.identity_op(sparse_input.ids)
        idx = sparse_input.indices
        embeddings = var.gather(unique_ids)
        sidx = sparse_input.sidx
        sseg = sparse_input.sseg
    else:
        with xdl.device(device, **device_attr):
            unique_ids, idx, sidx, sseg = xdl.unique(sparse_input.ids, sparse_input.segments, itype=DataType.int32)
        embeddings = var.gather(unique_ids)
        
    if statis_list is not None:
        assert labels is not None
        from xdl.python.training.training_utils import get_global_step
        global_step = get_global_step()
        statis_results = []
        for statis_var in statis_vars:
            statis_result = statis_var.statis(sparse_input.ids, idx, sparse_input.segments, sidx, sseg, labels, global_step.value)
            statis_results.append(statis_result)

    global _EMBEDDING_TENSOR
    _EMBEDDING_TENSOR[embeddings] = var

    import xdl.python.sparse_engine.embedding_ops as embedding_ops
    import numpy as np
    if combiner == 'sum':
        embeddings = embedding_ops.ksum(
            embeddings,
            idx,
            sparse_input.values,
            sparse_input.segments,
            sidx,
            sseg,
            device, **device_attr)
    elif combiner == 'mean':
        embeddings = embedding_ops.kmean(
            embeddings,
            idx,
            sparse_input.values,
            sparse_input.segments,
            sidx,
            sseg,
            device, **device_attr)
    elif combiner == 'tile':
        embeddings = embedding_ops.tile(
            embeddings,
            idx,
            np.array([], dtype=np.float32),
            #sparse_input.values,
            sparse_input.segments,
            length,
            reverse,
            device, **device_attr)
    else:
        raise Exception("Unrecognized combiner:" + str(combiner))

    if sparse_input.shape is not None and len(sparse_input.shape) > 0:
        embeddings.set_shape([sparse_input.shape[0], emb_dim]);

    emb_info = EmbeddingInfo(name, feature_dim, emb_dim, combiner, None, var, length, embeddings)
    set_embedding_info([var], emb_info)
    if statis_list is not None:
        return embeddings, statis_results
    return embeddings

#TODO delete. With sidx and sseg, following code will ill function.
def merged_embedding(name, sparse_inputs, initializer, emb_dim, feature_dim,
                     combiner='sum', vtype=VarType.Index, length=50, reverse=False,
                     batch_read=3000, feature_add_probability=1.0, cbf=0, device='CPU', **device_attr):
    """xdl embedding
       Args:
         name: name for embedding, will be used for declaring variable on ps-plus
         sparse_inputs: a list of sparse tensors represent input data
         initializer: intializer for the weights
         emb_dim: embedding dimension
         feature_dim: sparse input dimension, for pre-allocate memory
         combiner: reduce operator, support sum|mean
       Returns:
         a tensor represent embedding result
       Raises:
         None
    """
    import xdl.python.framework.variable as variable
    with variable.variable_info(batch_read=batch_read, save_ratio=feature_add_probability, bloom_filter=cbf):
        var = variable.Variable(name=name,
                                dtype=DataType.float,
                                shape=[feature_dim, emb_dim],
                                initializer=initializer,
                                vtype=vtype,
                                trainable = True)
    if isinstance(sparse_inputs, (list, tuple)):
        merged_sparse_inputs = merge_sparse(sparse_inputs)
        emb_dim *= len(sparse_inputs)
    else:
        assert(isinstance(sparse_inputs, MergedSparseTensor))
        merged_sparse_inputs = sparse_inputs
    if merged_sparse_inputs.has_unique_ids():
        unique_ids = merged_sparse_inputs.ids
        idx = merged_sparse_inputs.indices
        sidx = merged_sparse_inputs.sidx
        sseg = merged_sparse_inputs.sseg
    else:
        with xdl.device(device, **device_attr):
            unique_ids, idx, sidx, sseg = xdl.unique(ids, merged_sparse_inputs.groups, itype=DataType.int32)
    
    embeddings = var.gather(unique_ids)
    global _EMBEDDING_TENSOR
    _EMBEDDING_TENSOR[embeddings] = var
    import xdl.python.sparse_engine.embedding_ops as embedding_ops
    if combiner == 'sum':
        embeddings = embedding_ops.merged_ksum(
            embeddings,
            idx,
            merged_sparse_inputs.values,
            merged_sparse_inputs.segments,
            merged_sparse_inputs.groups,
            sidx,
            sseg,
            device, **device_attr)
    elif combiner == 'mean':
        embeddings = embedding_ops.merged_kmean(
            embeddings,
            idx,
            merged_sparse_inputs.values,
            merged_sparse_inputs.segments,
            merged_sparse_inputs.groups,
            sidx,
            sseg,
            device, **device_attr)
    elif combiner == 'tile':
        embeddings = embedding_ops.merged_tile(
            embeddings,
            idx,
            merged_sparse_inputs.values,
            merged_sparse_inputs.segments,
            merged_sparse_inputs.groups,
            length,
            reverse,
            device, **device_attr)
    else:
        raise Exception("Unrecognized combiner:" + str(combiner))

    emb_info = EmbeddingInfo(name, feature_dim, emb_dim, combiner, None, var, length, embeddings)
    set_embedding_info([var], emb_info)
    return embeddings


