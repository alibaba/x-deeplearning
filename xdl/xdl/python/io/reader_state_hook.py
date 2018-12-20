# Copyright (C) 2016-2018 Alibaba Group Holding Limited
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
import numpy as np
from xdl.python.framework.session import Hook
from xdl.python.framework.variable import Variable
from xdl.python.lib.datatype import DataType 
from xdl.python.ops.init_ops import Zeros
from xdl.python.lib.graph import execute
from xdl.python.utils.config import get_task_num, get_task_index

_STATE_SIZE = 256*1024

def _string_to_int8_array(s, capacity):
    arr = np.zeros((1, capacity), dtype=np.int8)
    for i in range(len(s)):
        arr[0][i] = ord(s[i])
    return arr

def _int8_array_to_string(arr):
    i = 0
    while i < len(arr):
        if arr[i] == 0:
            break
        i = i + 1
    return ''.join([chr(x) for x in arr[0:i]])

class ReaderStateHook(Hook):
    def __init__(self, data_io, save_interval=100):
        self._worker_num = get_task_num()
        self._worker_index = get_task_index()
        self._step = 0
        self._save_interval = save_interval
        self._data_io = data_io
        self._state_var = Variable(
            name=self._data_io.ds_name + "/reader_state", 
            shape=[self._worker_num, _STATE_SIZE],
            dtype=DataType.int8,
            initializer=Zeros(),
            trainable=False)

    def create_session(self):
        state_op = self._state_var.gather(
            np.array([self._worker_index], dtype=np.int32))
        state = execute(state_op)
        if (state == np.zeros((_STATE_SIZE), dtype=np.int8)).all():
            return
        else:
            self._data_io.restore_from_state(_int8_array_to_string(state[0]))

    def before_run(self, v):
        return []

    def after_run(self, v):
        self._step = self._step + 1
        if self._step % self._save_interval == 0:
            self._save_state()

    def _save_state(self):
        state = _string_to_int8_array(self._data_io.serialize_state(), 
                                      _STATE_SIZE)
        update_op = xdl.ps_sparse_assign_op(
            var_name=self._state_var.name,
            var_type=self._state_var.vtype,
            ids=np.array([self._worker_index], dtype=np.int32),
            values=state)
        execute(update_op)
        
