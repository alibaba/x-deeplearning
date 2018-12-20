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
import unittest
import numpy as np
from xdl.python.lib.datatype import *
from xdl.python.lib.graph import execute

class TestPsSynchronizerOps(unittest.TestCase):
    def test_all(self):
        as_enter_op = xdl.ps_asynchronize_enter_op(
                id = np.array(0, dtype=np.int32), 
                staleness = np.array(32, dtype=np.int32),
                worker_count = np.array(10, dtype=np.int32))
        s_enter_op = xdl.ps_synchronize_enter_op(
                id = np.array(0, dtype=np.int32), 
                worker_count = np.array(10, dtype=np.int32))
        s_leave_op = xdl.ps_synchronize_leave_op(id = np.array(0, dtype=np.int32))
        finish_op = xdl.ps_semi_synchronize_leave_op(id = np.array(0, dtype=np.int32))
        execute(as_enter_op)
        execute(s_enter_op)
        execute(s_leave_op)
        execute(finish_op)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestPsSynchronizerOps)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

