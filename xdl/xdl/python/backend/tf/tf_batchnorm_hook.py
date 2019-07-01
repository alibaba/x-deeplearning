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

import os
import sys

import tensorflow as tf
from xdl.python.utils.collections import *
from xdl.python.backend.model_scope import cur_model_scope
from tensorflow.python.keras.layers.normalization import BatchNormalization
import xdl

BATCHNORM_TENSORS = 'batchnorm_tf_tensors'

keras_layers_assgin_moving_average = BatchNormalization._assign_moving_average

def assign_moving_average(self, variable, value, momentum):
  var_mapping = get_collection(VAR_MAPPING, cur_model_scope())
  for x in var_mapping:
    if x[1] == variable:
      add_to_collection(BATCHNORM_TENSORS, (x[0], value, momentum), cur_model_scope())
  return keras_layers_assgin_moving_average(self, variable, value, momentum)

BatchNormalization._assign_moving_average = assign_moving_average

def get_batchnorm_tensors():
  res = get_collection(BATCHNORM_TENSORS, cur_model_scope())
  if res is None:
    return []
  return [v[1] for v in res]

def set_tf_output(output_tensors):
  res = get_collection(BATCHNORM_TENSORS, cur_model_scope())
  if res is None:
    return
  assert(len(res) == len(output_tensors))
  update_ops = []  
  for i in range(len(res)):
    update_op = xdl.ps_apply_moving_average_op(
      var_name = res[i][0].name, value = output_tensors[i], moment = res[i][2])
    update_ops.append(update_op)
  add_to_collection(UPDATE_OPS, update_ops)
