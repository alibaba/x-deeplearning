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

data = np.array([3,1,2,1,3,1,2,0,1,3,1,1,2,0,1,3,3,2,3,2,2,1,2,0])

class TestUnique(unittest.TestCase):
    def test_unique_cpu_1d(self):
        res_uniq = np.array([3,1,2,0])
        res_idx = np.array([0,1,2,1,0,1,2,3,1,0,1,1,
                            2,3,1,0,0,2,0,2,2,1,2,3])
        uniq, idx = xdl.unique(data, itype=DataType.int32)
        uniq, idx = xdl.execute([uniq, idx])
        self.assertTrue((uniq == res_uniq).all())
        self.assertTrue((idx == res_idx).all())

    def test_unique_gpu_1d(self):
        with xdl.device("GPU"):
            res_uniq = np.array([0,1,2,3])
            res_idx = np.array([3,1,2,1,3,1,2,0,1,3,1,1,
                                2,0,1,3,3,2,3,2,2,1,2,0])
            uniq, idx = xdl.unique(data, itype=DataType.int32)
            uniq, idx = xdl.execute([uniq, idx])
            self.assertTrue((uniq == res_uniq).all())
            self.assertTrue((idx == res_idx).all())

    def test_unique_cpu_2d(self):
        res_uniq = np.array([[3,1],[2,1],[2,0],[1,3],[1,1],[3,2]])
        res_idx = np.array([0,1,0,2,3,4,2,3,5,5,1,2])
        uniq, idx = xdl.unique(data.reshape((data.size/2, 2)),itype=DataType.int32)
        uniq, idx = xdl.execute([uniq, idx])
        self.assertTrue((uniq == res_uniq).all())
        self.assertTrue((idx == res_idx).all())

    def test_unique_gpu_2d(self):
        with xdl.device("GPU"):
            res_uniq = np.array([[1,1],[1,3],[2,0],[2,1],[3,1],[3,2]])
            res_idx = np.array([4,3,4,2,1,0,2,1,5,5,3,2])
            uniq, idx = xdl.unique(data.reshape((data.size/2, 2)),itype=DataType.int32)
            uniq, idx = xdl.execute([uniq, idx])
            self.assertTrue((uniq == res_uniq).all())
            self.assertTrue((idx == res_idx).all())

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestUnique)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
