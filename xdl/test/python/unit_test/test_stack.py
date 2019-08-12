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

class TestStack(unittest.TestCase):
    def test_stack_0(self):
      a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      b = np.array([[[11, 12], [13, 14]], [[15, 16], [17, 18]]])
      c = np.array([[[21, 22], [23, 24]], [[25, 26], [27, 28]]])
      r = xdl.stack([a, b, c], axis=0)
      r = xdl.execute(r)
      self.assertTrue(np.alltrue(r == np.stack([a, b, c], axis=0)))

    def test_stack_1(self):
      a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      b = np.array([[[11, 12], [13, 14]], [[15, 16], [17, 18]]])
      c = np.array([[[21, 22], [23, 24]], [[25, 26], [27, 28]]])
      r = xdl.stack([a, b, c], axis=1)
      r = xdl.execute(r)
      self.assertTrue(np.alltrue(r == np.stack([a, b, c], axis=1)))

    def test_stack_2(self):
      a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      b = np.array([[[11, 12], [13, 14]], [[15, 16], [17, 18]]])
      c = np.array([[[21, 22], [23, 24]], [[25, 26], [27, 28]]])
      r = xdl.stack([a, b, c], axis=2)
      r = xdl.execute(r)
      self.assertTrue(np.alltrue(r == np.stack([a, b, c], axis=2)))

    def test_stack_3(self):
      a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      b = np.array([[[11, 12], [13, 14]], [[15, 16], [17, 18]]])
      c = np.array([[[21, 22], [23, 24]], [[25, 26], [27, 28]]])
      r = xdl.stack([a, b, c], axis=3)
      r = xdl.execute(r)
      self.assertTrue(np.alltrue(r == np.stack([a, b, c], axis=3)))

    def test_stack_x1(self):
      a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      b = np.array([[[11, 12], [13, 14]], [[15, 16], [17, 18]]])
      c = np.array([[[21, 22], [23, 24]], [[25, 26], [27, 28]]])
      r = xdl.stack([a, b, c], axis=-1)
      r = xdl.execute(r)
      self.assertTrue(np.alltrue(r == np.stack([a, b, c], axis=-1)))

    def test_stack_x2(self):
      a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      b = np.array([[[11, 12], [13, 14]], [[15, 16], [17, 18]]])
      c = np.array([[[21, 22], [23, 24]], [[25, 26], [27, 28]]])
      r = xdl.stack([a, b, c], axis=-2)
      r = xdl.execute(r)
      self.assertTrue(np.alltrue(r == np.stack([a, b, c], axis=-2)))

    def test_stack_x3(self):
      a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      b = np.array([[[11, 12], [13, 14]], [[15, 16], [17, 18]]])
      c = np.array([[[21, 22], [23, 24]], [[25, 26], [27, 28]]])
      r = xdl.stack([a, b, c], axis=-3)
      r = xdl.execute(r)
      self.assertTrue(np.alltrue(r == np.stack([a, b, c], axis=-3)))

    def test_stack_x4(self):
      a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      b = np.array([[[11, 12], [13, 14]], [[15, 16], [17, 18]]])
      c = np.array([[[21, 22], [23, 24]], [[25, 26], [27, 28]]])
      r = xdl.stack([a, b, c], axis=-4)
      r = xdl.execute(r)
      self.assertTrue(np.alltrue(r == np.stack([a, b, c], axis=-4)))

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestStack)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

