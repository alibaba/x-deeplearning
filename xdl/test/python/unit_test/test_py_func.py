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

def func(a, b):
  return a * 2 + b

class TestPyFunc(unittest.TestCase):
  def test_py_func(self):
    op = xdl.py_func(func, [[10, 20, 30], [1, 2, 3]], [np.int64])
    rst = xdl.execute(op)
    self.assertTrue((rst == np.array([21, 42, 63])).all())

def suite():
  return unittest.TestLoader().loadTestsFromTestCase(TestPyFunc)

if __name__ == '__main__':
  unittest.TextTestRunner().run(suite())
