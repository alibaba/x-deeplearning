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

from xdl.python import pybind
from xdl.python.lib.tensorshape import TensorShape
from xdl.python.lib.datatype import DataType

def gen_attr(attr, name, attr_type):
  if isinstance(attr, pybind.AttrValue):
    if attr.attr_type != attr_type:
      raise ValueError("Attr[{0}] Should be type {1}".format(name, attr_type))
    return attr
  elif attr_type == pybind.AttrValue.Type.string:
    return gen_str(attr, name)
  elif attr_type == pybind.AttrValue.Type.bool:
    return gen_bool(attr, name)
  elif attr_type == pybind.AttrValue.Type.int:
    return gen_int(attr, name)
  elif attr_type == pybind.AttrValue.Type.float:
    return gen_float(attr, name)
  elif attr_type == pybind.AttrValue.Type.shape:
    return gen_shape(attr, name)
  elif attr_type == pybind.AttrValue.Type.type:
    return gen_type(attr, name)
  elif attr_type == pybind.AttrValue.Type.type_list:
    return gen_type_list(attr, name)

def gen_str(attr, name):
  if isinstance(attr, (str, unicode, bytes)):
    ret = pybind.AttrValue()
    ret.attr_type = pybind.AttrValue.Type.string
    ret.s = attr
    return ret
  else:
    raise ValueError("Cannot parse attr[{0}] type[{1}] to a str attr".format(name, attr.__class__))

def gen_bool(attr, name):
  if isinstance(attr, bool):
    ret = pybind.AttrValue()
    ret.attr_type = pybind.AttrValue.Type.bool
    ret.b = attr
    return ret
  else:
    raise ValueError("Cannot parse attr[{0}] type[{1}] to a bool attr".format(name, attr.__class__))

def gen_int(attr, name):
  if isinstance(attr, (int, long)):
    ret = pybind.AttrValue()
    ret.attr_type = pybind.AttrValue.Type.int
    ret.i = attr
    return ret
  else:
    raise ValueError("Cannot parse attr[{0}] type[{1}] to a int attr".format(name, attr.__class__))

def gen_float(attr, name):
  if isinstance(attr, float):
    ret = pybind.AttrValue()
    ret.attr_type = pybind.AttrValue.Type.float
    ret.f = attr
    return ret
  else:
    raise ValueError("Cannot parse attr[{0}] type[{1}] to a float attr".format(name, attr.__class__))

def gen_shape(attr, name):
  if isinstance(attr, (TensorShape, list, tuple)):
    ret = pybind.AttrValue()
    ret.attr_type = pybind.AttrValue.Type.shape
    ret.shape = pybind.TensorShape(pybind.SizeTVector(TensorShape(attr).dims()))
    return ret
  else:
    raise ValueError("Cannot parse attr[{0}] type[{1}] to a shape attr".format(name, attr.__class__))

def gen_type(attr, name):
  if isinstance(attr, DataType):
    ret = pybind.AttrValue()
    ret.attr_type = pybind.AttrValue.Type.type
    ret.type = attr
    return ret
  else:
    raise ValueError("Cannot parse attr[{0}] type[{1}] to a type attr".format(name, attr.__class__))

def gen_type_list(attr, name):
  if isinstance(attr, (list, tuple)):
    ret = pybind.AttrValue()
    ret.attr_type = pybind.AttrValue.Type.type_list
    ret.type_list = pybind.DataTypeVector(attr)
    return ret
  else:
    raise ValueError("Cannot parse attr[{0}] type[{1}] to a shape attr".format(name, attr.__class__))
