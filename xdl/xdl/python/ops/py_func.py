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
import traceback
import numpy as np
from xdl.python import pybind
from xdl.python.lib.error import XdlException
from xdl.python.lib.datatype import DataType
from xdl.python.lib.internal_ops import _py_func

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

def dtype_xdl_2_np(dtype):
    if dtype == DataType.float:
        return np.float32
    if dtype == DataType.double:
        return np.float64
    if dtype == DataType.int8:
        return np.int8
    if dtype == DataType.int16:
        return np.int16
    if dtype == DataType.int32:
        return np.int32
    if dtype == DataType.int64:
        return np.int64
    if dtype == DataType.bool:
        return np.bool
    raise ValueError("unsupported xdl dtype.")


def func_wrapper(func):
    def wrapper(inputs):
        inputs = [np.array(i, copy=False) for i in inputs]
        status = pybind.Status()
        rst_internal = []
        try:
            rst = func(*inputs)
            if rst is None:
                rst = []
            if isinstance(rst, np.ndarray):
                rst = [rst]
            for item in rst:
                tensor = pybind.PyFuncResult.TensorBuffer()
                tensor.buf = item.tostring()
                tensor.shape = pybind.SizeTVector(list(item.shape))
                tensor.type = dtype_np_2_xdl(item.dtype)
                rst_internal.append(tensor)
        except XdlException as exc:
            status = pybind.Status(exc.code,
                                   exc.msg + "\n" + traceback.format_exc())
        except Exception as exc:
            status = pybind.Status(pybind.Status.ErrorCode.Internal,
                                   repr(exc) + "\n" + traceback.format_exc())
        result = pybind.PyFuncResult()
        result.status = status
        result.result = pybind.PyFuncResult.TensorBufferVector(rst_internal)
        return result
    return wrapper

def py_func(func, input, output_type = None):
  if output_type is None:
    output_type = []
  if isinstance(output_type, (np.dtype, DataType)):
    output_type = [output_type]
  real_output_type = []
  for i in output_type:
    if isinstance(i, type):
      real_output_type.append(dtype_np_2_xdl(i))
    elif isinstance(i, DataType):
      real_output_type.append(i)
    else:
      raise ValueError("cannot parse output_type")
  handle = pybind.object_handle(func_wrapper(func))
  return _py_func(input, handle=handle, output_type=real_output_type)
