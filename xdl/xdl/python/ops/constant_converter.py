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

import numpy as np
from xdl.python.lib.internal_ops import _constant
from xdl.python.lib.tensor import register_converter
from xdl.python.lib.datatype import DataType

def dtype_np_2_xdl(dtype):
  if dtype == np.float32:
    return DataType.float
  if dtype == np.float64:
    return DataType.double
  if dtype == np.int8:
    return DataType.int8
  if dtype == np.int16:
    return DataType.int16
  if dtype == np.int32:
    return DataType.int32
  if dtype == np.int64:
    return DataType.int64
  if dtype == np.bool:
    return DataType.bool
  raise ValueError("unsupported numpy dtype.")

def converter(t):
  npt = np.array(t)
  shape = list(npt.shape)
  dtype = dtype_np_2_xdl(npt.dtype)
  b = str(npt.data)
  return _constant(dtype, shape, b)

register_converter(int)(converter)
register_converter(float)(converter)
register_converter(list)(converter)
register_converter(np.ndarray)(converter)
