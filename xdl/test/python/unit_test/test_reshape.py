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

class TestReshape(unittest.TestCase):
    def test_reshape(self):
      value = np.array([[1, 2], [3, 4]])
      a = xdl.reshape(value, [4])
      a = xdl.execute(a)
      self.assertTrue(np.alltrue(a == [1, 2, 3, 4]))

    def test_reshape_x(self):
      value = np.array([[1, 2], [3, 4]])
      a = xdl.reshape(value, [1, -1])
      a = xdl.execute(a)
      self.assertTrue(np.alltrue(a == [[1, 2, 3, 4]]))

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestReshape)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
