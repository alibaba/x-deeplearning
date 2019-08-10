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
from xdl.python.lib.tensor import Tensor

class Op(object):
  def __init__(self, inputs, attrs, name, op, device_name='CPU'):
    self._inputs = inputs
    self._outputs = []
    self._attrs = attrs
    self._name = name
    self._op = op
    self._device_name = device_name

  def set_outputs(self, outputs):
    self._outputs = outputs

  def output_define(self, k):
    return self.name + ":" + str(k)

  def depend(self):
    return Tensor('^' + self._name, DataType.int8, self)

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    return self._outputs

  @property
  def attrs(self):
    return self._attrs

  @property
  def name(self):
    return self._name

  @property
  def op(self):
    return self._op

  @property
  def device_name(self):
    return self._device_name

