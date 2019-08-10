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

from xdl.python.backend.model_scope import cur_model_scope

_VAR_AND_GRADS = {}
_GEAR_GRADS = {}

def set_gradients(var_names, gradients, scope):
  global _VAR_AND_GRADS
  if scope not in _VAR_AND_GRADS:
    _VAR_AND_GRADS[scope] = {}
  var_and_grads = _VAR_AND_GRADS[scope]
  for i in range(len(var_names)):
    if var_names[i] in var_and_grads:
      if not isinstance(var_and_grads[var_names[i]], list):
        var_and_grads[var_names[i]] = [var_and_grads[var_names[i]]]
      var_and_grads[var_names[i]].append(gradients[i])
    else:
      var_and_grads[var_names[i]] = gradients[i]    

def get_gradient(var_name, scope=None):
  scope = scope if scope is not None else cur_model_scope()
  global _VAR_AND_GRADS    
  if scope not in _VAR_AND_GRADS or \
        var_name not in _VAR_AND_GRADS[scope]:
    return None
  return _VAR_AND_GRADS[scope][var_name]

def get_gradients():
  global _VAR_AND_GRADS    
  return _VAR_AND_GRADS    

def reset_gradients():
  global _VAR_AND_GRADS
  _VAR_AND_GRADS = {}

def set_gear_gradient(x, grad):
  global _GEAR_GRADS
  _GEAR_GRADS[x] = grad

def get_gear_gradient(x):
  global _GEAR_GRADS
  return _GEAR_GRADS[x]

_VAR_MAPPING = {}
def add_var_mapping(name, var, scope=None):
  global _VAR_MAPPING
  scope = scope if scope is not None else cur_model_scope()
  if scope not in _VAR_MAPPING:
    _VAR_MAPPING[scope] = {}
  if name in _VAR_MAPPING[scope]:
    raise 'duplicate key:', name
  _VAR_MAPPING[scope][name] = var

def get_var_mapping(key, scope=None):
  global _VAR_MAPPING
  scope = scope if scope is not None else cur_model_scope()
  if scope not in _VAR_MAPPING:
    return None
  var_mapping = _VAR_MAPPING[scope]
  if key in var_mapping:
    return var_mapping[key]
  return None
  
