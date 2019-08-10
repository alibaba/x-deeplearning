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

class TestDenseApplyAdamOp(unittest.TestCase):
    def test_all(self):
        var = xdl.Variable(name="w", dtype=DataType.float, shape=[4], initializer=xdl.Ones())
        execute(xdl.variable_registers())
        execute(xdl.global_initializers())
        op = xdl.ps_dense_apply_adam_op(
            beta1=np.array(0.9, dtype=np.float),
            beta2=np.array(0.999, dtype=np.float),
            epsilon=np.array(1e-08, dtype=np.float),
            learning_rate=np.array(0.1, dtype=np.float),
            grad=np.array([1,2,3,4], dtype=np.float32),
            lr_decay=True,
            var_name="w",
            var_type="index")
        execute(op)
        ret = execute(var.value)
        print(ret)
        self.assertTrue((ret == np.array([0.90000004,0.90000004,0.90000004,0.90000004], dtype=np.float32)).all())
        execute(op)
        ret = execute(var.value)
        self.assertTrue((ret == np.array([0.8000001,0.8000001,0.8000001,0.8], dtype=np.float32)).all())

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestDenseApplyAdamOp)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
