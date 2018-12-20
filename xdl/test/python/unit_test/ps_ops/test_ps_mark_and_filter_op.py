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
from xdl.python.framework.variable import VarType

class TestPsMarkAndFilterOp(unittest.TestCase):
    def test_all(self):
        var = xdl.Variable(name="w", dtype=DataType.int64, shape=[4,8], 
                           vtype = VarType.Hash, initializer=xdl.Ones())
        execute(xdl.variable_registers())
        execute(xdl.global_initializers())
        mark_op = xdl.ps_mark_op(
          var_name = "w", 
          ids = np.array([[10,10],[10,10],[12,12]], dtype=np.int64),
          pattern = "g",
          i = 12)
        execute(mark_op)
        filter_op = xdl.ps_filter_op(
          var_name = "w",
          pattern = "i==g",
          i = 12, d = 0.5)
        execute(filter_op)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestPsMarkAndFilterOp)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

