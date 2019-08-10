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
import time
import numpy as np
from xdl.python.lib.graph import execute

def _string_to_int8(src):
    return np.array([ord(ch) for ch in src], dtype=np.int8)

def convert_ps_variable(var_list, ckpt_dir, output_dir, with_slots=False):
    var_str_list = ','.join(var_list)
    op = xdl.ps_convert_ckpt_variable_op(
        variables = _string_to_int8(var_str_list),
        checkpoint_dir=_string_to_int8(ckpt_dir),
        output_dir=_string_to_int8(output_dir),
        with_slots=with_slots)
    execute(op)

_BARRIER_NAME_2_ID = {}
_BARRIER_TOKEN = int(time.time())

def barrier(task_id, task_num, name=None):
    global _BARRIER_NAME_2_ID
    global _BARRIER_TOKEN
    if name is None:
        name = '_default_barrier_'
    if name not in _BARRIER_NAME_2_ID:
        _BARRIER_NAME_2_ID[name] = len(_BARRIER_NAME_2_ID)
    barrier_id = _BARRIER_NAME_2_ID[name]
    barrier_op = xdl.worker_barrier_v2op(
        barrier_id=np.array(barrier_id, dtype=np.int32), 
        task_id=np.array(task_id, dtype=np.int32),
        task_num=np.array(task_num, dtype=np.int32), 
        token=np.array(_BARRIER_TOKEN, dtype=np.int32))
    execute(barrier_op)

def barrier_op_v2(task_id, task_num, name=None):
    global _BARRIER_NAME_2_ID
    global _BARRIER_TOKEN
    if name is None:
        name = '_default_barrier_'
    if name not in _BARRIER_NAME_2_ID:
        _BARRIER_NAME_2_ID[name] = len(_BARRIER_NAME_2_ID)
    barrier_id = _BARRIER_NAME_2_ID[name]
    barrier_op = xdl.worker_barrier_v2op(
        barrier_id=np.array(barrier_id, dtype=np.int32), 
        task_id=np.array(task_id, dtype=np.int32),
        task_num=np.array(task_num, dtype=np.int32), 
        token=np.array(_BARRIER_TOKEN, dtype=np.int32))
    return barrier_op
