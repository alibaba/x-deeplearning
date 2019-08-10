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
import unittest
import numpy as np
from xdl.python.lib.datatype import *
from xdl.python.lib.graph import execute

class TestSparseApplyFtrlOp(unittest.TestCase):
    def test_all(self):
        var = xdl.Variable(name="w", dtype=DataType.float, shape=[4,1], initializer=xdl.Ones())
        execute(xdl.variable_registers())
        execute(xdl.global_initializers())
        op = xdl.ps_sparse_apply_ftrl_op(
            learning_rate=np.array(0.1, dtype=np.float),
            learning_rate_power=np.array(-0.5, dtype=np.float),
            initial_accumulator_value=np.array(0.1, dtype=np.float),
            l1_reg=np.array(0, dtype=np.float),
            l2_reg=np.array(0, dtype=np.float),
            grad=np.array([[1],[2]], dtype=np.float32),
            indices=np.array([1,2], dtype=np.int32),
            var_name="w",
            var_type="index")
        execute(op)
        ret = execute(var.value)
        self.assertTrue((ret == np.array([[1],[0.6031424],[0.7450533],[1]], dtype=np.float32)).all())
        execute(op)
        ret = execute(var.value)
        self.assertTrue((ret == np.array([[1],[0.5341358],[0.6747804],[1]], dtype=np.float32)).all())

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestSparseApplyFtrlOp)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
