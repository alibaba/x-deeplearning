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

from xdl.python.lib import internal_ops
import inspect

def hash_filter(var_name, fdef, func_name, payload={}):
  func_def = open(fdef).read()
  exec(func_def)
  f = locals()[func_name]
  f_args = inspect.getargspec(f)
  if f_args.keywords is not None or f_args.varargs is not None:
    raise ValueError("function should not have varargs or keywords")
  f_args = f_args.args
  payload_name = payload.keys()
  payload_value = [payload[i] for i in payload_name]
  return internal_ops.ps_filter_op(payload_value, var_name, func_def, func_name, ";".join(f_args), ";".join(payload_name))

def hash_slot_filter(var_name, fdef, func_name, slot_name, slot_size, payload={}):
  func_def = open(fdef).read()
  exec(func_def)
  f = locals()[func_name]
  f_args = inspect.getargspec(f)
  if f_args.keywords is not None or f_args.varargs is not None:
    raise ValueError("function should not have varargs or keywords")
  f_args = f_args.args
  payload_name = payload.keys()
  payload_value = [payload[i] for i in payload_name]
  return internal_ops.ps_slot_filter_op(payload_value, var_name, func_def, func_name, ";".join(f_args), ";".join(payload_name), slot_name, slot_size)
