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

"""collections is used to store some global informations."""

GLOBAL_COLLECTION = {}

GLOBAL_VARIABLES = 'global_variables'
TRAINABLE_VARIABLES = 'trainable_variables'
GLOBAL_INITIALIZERS = 'global_initializers'
REGULARIZER_LOSS = 'regularizer_loss'
VAR_MAPPING = 'var_mapping'
BACKPROP_VARS = 'backprop_vars'
VAR_REGISTERS = 'var_registers'
READER_HOOKS = 'reader_hooks'
TF_VAR_DICT = 'tf_var_dict'
GEAR_GRAD = 'gear_grad'
MXNET_BN_STATISTIC = 'mxnet_bn_statistic'
BN_STATISTIC = 'bn_statistic'
UPDATE_OPS = 'update_ops'
BACKEND_DEVICE_TYPE = 'backend_device_type'

def get_scopes(scope):
  scopes = []
  if scope is None:
    scopes.append(cur_model_scope())
  elif isinstance(scope, (list, tuple)):
    scopes = list(scope)
  else:
    scopes = [scope]
  return list(set(scopes))

def name_with_scope(name, scope):
  return name if scope == '' else scope + '/' + name

def add_to_collection(name, value, scope=None):
  scopes = get_scopes(scope)
  for scope in scopes:
    new_name = name_with_scope(name, scope)
    global GLOBAL_COLLECTION
    if new_name not in GLOBAL_COLLECTION:
      GLOBAL_COLLECTION[new_name] = []        
    if isinstance(value, list):
      GLOBAL_COLLECTION[new_name].extend(value)
    else:
      GLOBAL_COLLECTION[new_name].append(value)

def add_to_collections(names, value, scope=None):
  scopes = get_scopes(scope)
  for scope in scopes:
    for name in names:
      add_to_collection(name, value, scope)

def get_collection(name, scope=None):
  scopes = get_scopes(scope)
  result = []
  for scope in scopes:
    new_name = name_with_scope(name, scope)
    global GLOBAL_COLLECTION
    if new_name in GLOBAL_COLLECTION:
      result.extend(GLOBAL_COLLECTION[new_name])
  if len(result) == 0:
    return None
  return result

def delete_collection(name, scope=None):
  scopes = get_scopes(scope)
  for scope in scopes:
    new_name = name_with_scope(name, scope)
    global GLOBAL_COLLECTION
    if new_name in GLOBAL_COLLECTION:
      del GLOBAL_COLLECTION[new_name]

def delete_collections(names, scope=None):
  scopes = get_scopes(scope)
  for scope in scopes:
    for name in names:
      delete_collection(name,  scope)

def reset_collections():
  global GLOBAL_COLLECTION
  GLOBAL_COLLECTION = {}    


