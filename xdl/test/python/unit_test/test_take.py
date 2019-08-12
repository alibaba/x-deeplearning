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

comm = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],dtype=np.float)
indicator = np.array([0,0,1,1,2],dtype=np.int32)

class TestTake(unittest.TestCase):
    def test_cpu(self):
        out = xdl.take_op(comm, indicator)
        out = xdl.execute(out)
        res = np.array([[0.1,0.2,0.3],[0.1,0.2,0.3],[0.4,0.5,0.6],
                       [0.4,0.5,0.6],[0.7,0.8,0.9]],dtype=np.float)
        self.assertTrue(np.allclose(out, res))

    def test_gpu(self):
        with xdl.device("GPU"):
            out = xdl.take_op(comm, indicator)
            out = xdl.execute(out)
            res = np.array([[0.1,0.2,0.3],[0.1,0.2,0.3],[0.4,0.5,0.6],
                           [0.4,0.5,0.6],[0.7,0.8,0.9]],dtype=np.float)
            self.assertTrue(np.allclose(out, res))

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestTake)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
