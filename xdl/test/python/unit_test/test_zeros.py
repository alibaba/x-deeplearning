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

class TestZeros(unittest.TestCase):
    def test_zeros_0(self):
      a = xdl.zeros(np.array([], np.int64), xdl.DataType.float)
      a = xdl.execute(a)
      self.assertTrue(np.alltrue(a == 0))
      self.assertTrue(a.shape == ())
      self.assertTrue(a.dtype == np.float32)

    def test_zeros_1(self):
      a = xdl.zeros([100], xdl.DataType.float)
      a = xdl.execute(a)
      self.assertTrue(np.alltrue(a == 0))
      self.assertTrue(a.shape == (100,))
      self.assertTrue(a.dtype == np.float32)

    def test_zeros_int64(self):
      a = xdl.zeros([100], xdl.DataType.int64)
      a = xdl.execute(a)
      self.assertTrue(np.alltrue(a == 0))
      self.assertTrue(a.shape == (100,))
      self.assertTrue(a.dtype == np.int64)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestZeros)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
