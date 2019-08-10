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
import numpy as np
from xdl.python.framework.session import Hook
from xdl.python.framework.variable import Variable
from xdl.python.lib.datatype import DataType
from xdl.python.ops.init_ops import Zeros
from xdl.python.lib.graph import execute
from xdl.python.utils.config import get_task_num, get_task_index
from xdl.python.proto import io_state_pb2 

class GlobalReaderStateHook(Hook):
    def __init__(self, data_io, save_interval=100):
        super(GlobalReaderStateHook, self).__init__()
        self._worker_num = get_task_num()
        self._worker_index = get_task_index()
        self._step = 0
        self._save_interval = save_interval
        self._data_io = data_io
        self._state = io_state_pb2.DSState()
        self._state.ds_name = data_io.ds_name
        self._state.epochs = 0

    def create_session(self):
        self._data_io.shutdown(True)
        self._data_io.restore_from_state(str(self._state))
        self._data_io.startup()

    def before_run(self, v):
        return []

    def after_run(self, v):
        self._step = self._step + 1
        if self._step % self._save_interval == 0:
            self._data_io.serialize_state()
