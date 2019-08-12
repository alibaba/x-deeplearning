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

class TestSplit(unittest.TestCase):
    def test_split_0_dim(self):
      value = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])
      num_or_size_splits = np.array([1, 3])
      a, b = xdl.split(value, num_or_size_splits, 0, 2)
      a, b = xdl.execute([a, b])
      self.assertTrue(np.alltrue(a == [[10, 20]]))
      self.assertTrue(np.alltrue(b == [[30, 40], [50, 60], [70, 80]]))

    def test_split_0_dim_x(self):
      value = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])
      num_or_size_splits = np.array(2)
      a, b = xdl.split(value, num_or_size_splits, 0, 2)
      a, b = xdl.execute([a, b])
      self.assertTrue(np.alltrue(a == [[10, 20], [30, 40]]))
      self.assertTrue(np.alltrue(b == [[50, 60], [70, 80]]))

    def test_split_neg_dim(self):
      value = np.array([[10, 20, 1], [30, 40, 1], [50, 60, 1], [70, 80, 1]])
      num_or_size_splits = np.array([1, 2])
      a, b = xdl.split(value, num_or_size_splits, -1, 2)
      a, b = xdl.execute([a, b])
      self.assertTrue(np.alltrue(a == [[10], [30], [50], [70]]))
      self.assertTrue(np.alltrue(b == [[20, 1], [40, 1], [60, 1], [80, 1]]))

    def test_split_neg_dim_x(self):
      value = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])
      num_or_size_splits = np.array(2)
      a, b = xdl.split(value, num_or_size_splits, -1, 2)
      a, b = xdl.execute([a, b])
      self.assertTrue(np.alltrue(a == [[10], [30], [50], [70]]))
      self.assertTrue(np.alltrue(b == [[20], [40], [60], [80]]))

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestSplit)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
