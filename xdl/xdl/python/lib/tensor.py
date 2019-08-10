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

class Tensor(object):
  def __init__(self, define, dtype, op):
    self._define = define
    self._dtype = dtype
    self._op = op
    self._shape = None
    self._name = None

  def __repr__(self):
    return '<Tensor "{0}" dtype={1}>'.format(self._define, repr(self._dtype))

  def __str__(self):
    return '<Tensor "{0}" dtype={1}>'.format(self._define, repr(self._dtype))

  @property
  def define(self):
    return self._define

  @property
  def dtype(self):
    return self._dtype

  @property
  def op(self):
    return self._op

  @property
  def shape(self):
    return self._shape

  @property
  def name(self):
    return self._name

  def set_shape(self, shape):
    self._shape = shape

  def set_name(self, name):
    self._name = name

class TensorConverter(object):
  _convert_map = []

  @staticmethod
  def convert_to_tensor(data):
    for priority, t, fn in TensorConverter._convert_map:
      if isinstance(data, t):
        return fn(data)
    raise ValueError("Cannot convert to tensor: " + repr(data))

  @staticmethod
  def register_converter(t, priority=0):
    def wrapper(fn):
      convert_map = TensorConverter._convert_map
      convert_map.append((-priority, t, fn))
      convert_map.sort()
      return fn
    return wrapper

def convert_to_tensor(t):
  return TensorConverter.convert_to_tensor(t)

def register_converter(t, priority=0):
  return TensorConverter.register_converter(t, priority)

@register_converter(Tensor, -65536)
def _tensor_to_tensor(t):
  return t
