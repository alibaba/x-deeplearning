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

class TestAuc(unittest.TestCase):
    def test_auc(self):
        labels = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.float32)
        predicts = np.array([0.7, 0.2, 0.6, 0.8, 0.1, 0.2, 0.6, 0.9, 0.1, 0.1], dtype=np.float32)
        res = xdl.auc(predicts, labels)
        execute(xdl.variable_registers())
        execute(xdl.global_initializers())
        res = xdl.execute(res)
        print res

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestAuc)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
