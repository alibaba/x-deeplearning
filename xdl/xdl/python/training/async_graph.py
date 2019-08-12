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

from xdl.python.lib.datatype import DataType
from xdl.python.sparse_engine.base import SparseTensor
from xdl.python.lib.graph import current_graph, Graph
import xdl

class Offset():
  global_offset = 0
  def __init__(self, type_name, name, obj):
    self._type = type_name
    self._name = name
    self._offset = self.global_offset
    self._length = 0
    if type_name == "sparse_list":
      self._length = len(obj)
    elif type_name == "sparse":
      if obj.indices is not None:
        Offset.global_offset += 6
        self._has_unique_id = True
      else:
        Offset.global_offset += 3
        self._has_unique_id = False
    elif type_name == "merged_sparse":
      if obj.indices is not None:
        Offset.global_offset += 7
        self._has_unique_id = True
      else:
        Offset.global_offset += 4
        self._has_unique_id = False
    elif type_name == "dense":
      Offset.global_offset += 1
    else:
      raise ValueError("invalid type " + str(type_name))
  @property
  def offset(self):
    return self._offset
  @property
  def type(self):
    return self._type
  @property
  def name(self):
    return self._name
  @property
  def length(self):
    return self._length
  @property
  def has_unique_id(self):
    return self._has_unique_id

class AsyncGraph():
  def __init__(self):
    self._enqueue_tensors = []
    self._names = []
    self._output_types = []
    self._offsets_map = {}

  def _regist_in_map(self, name, offset_obj):
    if name in self._offsets_map:
      raise ValueError(name + " encountered again")
    else:
      self._offsets_map[name] = offset_obj

  def enqueue_sparse(self, name, sparse_tensor, regist=True):
    self._enqueue_tensors.append(sparse_tensor.ids)
    self._enqueue_tensors.append(sparse_tensor.values)
    self._enqueue_tensors.append(sparse_tensor.segments)
    self._names.append(name + "_ids")
    self._names.append(name + "_values")
    self._names.append(name + "_segments")
    self._output_types.append(DataType.int64)
    self._output_types.append(DataType.float)
    self._output_types.append(DataType.int32)
    if sparse_tensor.indices is not None:
      self._enqueue_tensors.append(sparse_tensor.indices)
      self._enqueue_tensors.append(sparse_tensor.sidx)
      self._enqueue_tensors.append(sparse_tensor.sseg)
      self._names.append(name + "_indices")
      self._names.append(name + "_sidx")
      self._names.append(name + "_sseg")
      self._output_types.append(DataType.int32)
      self._output_types.append(DataType.int32)
      self._output_types.append(DataType.int32)
    if regist:
      self._regist_in_map(name, Offset("sparse", name, sparse_tensor))

  def enqueue_merged_sparse(self, name, sparse_tensor, regist=True):
    self._enqueue_tensors.append(sparse_tensor.ids)
    self._enqueue_tensors.append(sparse_tensor.values)
    self._enqueue_tensors.append(sparse_tensor.segments)
    self._enqueue_tensors.append(sparse_tensor.groups)    
    self._names.append(name + "_ids")
    self._names.append(name + "_values")
    self._names.append(name + "_segments")
    self._names.append(name + "_groups")    
    self._output_types.append(DataType.int64)
    self._output_types.append(DataType.float)
    self._output_types.append(DataType.int32)
    self._output_types.append(DataType.int32)    
    if sparse_tensor.indices is not None:
      self._enqueue_tensors.append(sparse_tensor.indices)
      self._enqueue_tensors.append(sparse_tensor.sidx)
      self._enqueue_tensors.append(sparse_tensor.sseg)
      self._names.append(name + "_indices")
      self._names.append(name + "_sidx")
      self._names.append(name + "_sseg")
      self._output_types.append(DataType.int32)
      self._output_types.append(DataType.int32)
      self._output_types.append(DataType.int32)
    if regist:
      self._regist_in_map(name, Offset("merged_sparse", name, sparse_tensor))

  def enqueue_sparse_list(self, name, sparse_list):
    for i in range(len(sparse_list)):
      sparse_tensor = sparse_list[i]
      unique_ids, idx, sidx, sseg = xdl.unique(sparse_tensor.ids, sparse_tensor.segments, itype=DataType.int32)
      sparse_tensor._ids = unique_ids
      sparse_tensor._indices = idx
      self.enqueue_sparse("__" + name + "_" + str(i), sparse_tensor)
    self._regist_in_map(name, Offset("sparse_list", name, sparse_list))

  def enqueue_dense(self, name, dense_tensor, regist=True):
    self._enqueue_tensors.append(dense_tensor)
    self._names.append(name)
    self._output_types.append(DataType.float)
    if regist:
      self._regist_in_map(name, Offset("dense", name, dense_tensor))

  def dequeue(self):
    tensors = xdl.dequeue_op(types=self._output_types)
    ret = {}
    for name in self._offsets_map.keys():
      if name.startswith("__"):
        continue
      offset = self._offsets_map[name]
      if offset.type == "sparse":
        ret[name] = self._dequeue_sparse(tensors, offset)
      elif offset.type == "merged_sparse":
        ret[name] = self._dequeue_merged_sparse(tensors, offset)        
      elif offset.type == "sparse_list":
        ret[name] = self._dequeue_sparse_list(tensors, offset, offset.length)
      elif offset.type == "dense":
        ret[name] = tensors[offset.offset]
    print("--------AsyncGraph--------")
    print(ret.keys())
    print("--------AsyncGraph--------")  
    return ret

  def _dequeue_sparse(self, tensors, offset):
    offset_idx = offset.offset
    if offset.has_unique_id:
      return xdl.SparseTensor(tensors[offset_idx], tensors[offset_idx + 1], tensors[offset_idx + 2], tensors[offset_idx + 3], tensors[offset_idx + 4], tensors[offset_idx + 5])
    else:
      return xdl.SparseTensor(tensors[offset_idx], tensors[offset_idx + 1], tensors[offset_idx + 2])

  def _dequeue_merged_sparse(self, tensors, offset):
    offset_idx = offset.offset
    if offset.has_unique_id:
      return xdl.MergedSparseTensor(tensors[offset_idx], tensors[offset_idx + 1], tensors[offset_idx + 2], tensors[offset_idx + 3], tensors[offset_idx + 4], tensors[offset_idx + 5], tensors[offset_idx + 6])
    else:
      return xdl.MergedSparseTensor(tensors[offset_idx], tensors[offset_idx + 1], tensors[offset_idx + 2], tensors[offset_idx + 3])

  def _dequeue_sparse_list(self, tensors, offset, length):
    ret = []
    for i in range(length):
      key = "__" + offset.name + "_" + str(i)
      tmp_offset = self._offsets_map[key]
      ret.append(self._dequeue_sparse(tensors, tmp_offset))
    return ret

  def enqueue_start(self):
    enqueue = xdl.enqueue_op(tensors=self._enqueue_tensors, names=";".join(self._names))
    current_graph().execute_loop(enqueue)
    xdl.Graph._current_graph.pop()

  def __enter__(self):
    return self    

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.enqueue_start()
    xdl.Graph._current_graph.append(xdl.Graph())

