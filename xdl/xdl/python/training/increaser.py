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

import os
import numpy as np
from xdl.python.framework.session import Hook
from xdl.python.training.training_utils import get_global_step
from xdl.python.lib.graph import execute


def _string_to_int8(src):
    return np.array([ord(ch) for ch in src], dtype=np.int8)


class DenseIncreaser(object):
    def increase(self, inc_version):
        execute(self.increase_op(inc_version))
    def increase_op(self, inc_version):
        return xdl.ps_streaming_trigger_dense_op(_string_to_int8(inc_version))

class DenseIncreaseStreamHook(Hook):
    def __init__(self, save_interval_step, is_training=True):
        super(DenseIncreaseStreamHook, self).__init__()
        self._global_step = get_global_step()
        self._save_interval = save_interval_step
        self._saver = DenseIncreaser()
        self._is_training = is_training
        self._last_save_step = 0

    def before_run(self, v):
        if (self._is_training):
            return self._global_step.value

    def after_run(self, v):
        global_step = v[0] if isinstance(v, list) else v
        if global_step - self._last_save_step >= self._save_interval:
            inc_version = self._create_version(global_step)
            print('increase stream at global_step[%d], increase version[%s]' % (global_step, inc_version))
            self._saver.increase(inc_version)
            self._last_save_step = global_step

    def _create_version(self, global_step):
        return "inc-{:.>20}".format(global_step)


class HashIncreaser(object):
    def increase(self, inc_version):
        execute(self.increase_op(inc_version))
    def increase_op(self, inc_version):
        return xdl.ps_streaming_trigger_hash_op(_string_to_int8(inc_version))

class HashIncreaseStreamHook(Hook):
    def __init__(self, save_interval_step, is_training=True):
        super(HashIncreaseStreamHook, self).__init__()
        self._global_step = get_global_step()
        self._save_interval = save_interval_step
        self._saver = HashIncreaser()
        self._is_training = is_training
        self._last_save_step = 0

    def before_run(self, v):
        if (self._is_training):
            return self._global_step.value

    def after_run(self, v):
        global_step = v[0] if isinstance(v, list) else v
        if global_step - self._last_save_step >= self._save_interval:
            inc_version = self._create_version(global_step)
            print('increase stream at global_step[%d], increase version[%s]' % (global_step, inc_version))
            self._saver.increase(inc_version)
            self._last_save_step = global_step

    def _create_version(self, global_step):
        return "inc-{:.>20}".format(global_step)


class SparseIncreaser(object):
    def increase(self, inc_version):
        execute(self.increase_op(inc_version))
    def increase_op(self, inc_version):
        return xdl.ps_streaming_trigger_sparse_op(_string_to_int8(inc_version))

class SparseIncreaseStreamHook(Hook):
    def __init__(self, save_interval_step, is_training=True):
        super(SparseIncreaseStreamHook, self).__init__()
        self._global_step = get_global_step()
        self._save_interval = save_interval_step
        self._saver = SparseIncreaser()
        self._is_training = is_training
        self._last_save_step = 0

    def before_run(self, v):
        if (self._is_training):
            return self._global_step.value

    def after_run(self, v):
        global_step = v[0] if isinstance(v, list) else v
        if global_step - self._last_save_step >= self._save_interval:
            inc_version = self._create_version(global_step)
            print('increase stream at global_step[%d], increase version[%s]' % (global_step, inc_version))
            self._saver.increase(inc_version)
            self._last_save_step = global_step

    def _create_version(self, global_step):
        return "inc-{:.>20}".format(global_step)
