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
from xdl.python.ops.constant_converter import *

data = np.array([3,1,2,1,3,1,2,0,1,3,1,1,2,0,1,3,3,2,3,2,2,1,2,0])
                   
class TestUnique(unittest.TestCase):
    def test_unique_cpu_1d(self):
        res_uniq = np.array([0,2,1,3])
        res_idx = np.array([3,2,1,2,3,2,1,0,2,3,2,2,
                            1,0,2,3,3,1,3,1,1,2,1,0])
        res_sidx = np.array([2,6,10,0,2,6,7,8,9,10,0,1,2,3,4,5,6,9,0,1,3,6,7,8])
        res_sseg = np.array([3,10,18,24])
        
        segment = np.array([3,5,8,10,11,12,16,18,20,22,24], np.int32)
        uniq, idx, sidx, sseg = xdl.unique(input=data, segment=segment, itype=DataType.int32)
        uniq, idx, sidx, sseg = xdl.execute([uniq, idx, sidx, sseg])
        self.assertTrue((uniq == res_uniq).all())
        self.assertTrue((idx == res_idx).all())
        self.assertTrue((sidx == res_sidx).all())
        self.assertTrue((sseg == res_sseg).all())

        segment = np.array([3,5,8,10,11,12,16,18,20,22,24], np.int64)
        uniq, idx, sidx, sseg = xdl.unique(input=data, segment=segment, itype=DataType.int64)
        uniq, idx, sidx, sseg = xdl.execute([uniq, idx, sidx, sseg])
        self.assertTrue((uniq == res_uniq).all())
        self.assertTrue((idx == res_idx).all())
        self.assertTrue((sidx == res_sidx).all())
        self.assertTrue((sseg == res_sseg).all())        

        segment2 = np.array([3,5,5,8,10,11,12,12,16,18,20,22,24], np.int32)
        uniq, idx, sidx, sseg = xdl.unique(input=data, segment=segment2, itype=DataType.int32)
        uniq, idx, sidx, sseg = xdl.execute([uniq, idx, sidx, sseg])
        res_sidx = np.array([3,8,12,0,3,8,9,10,11,12,0,1,3,4,5,6,8,11,0,1,4,8,9,10])
        self.assertTrue((uniq == res_uniq).all())
        self.assertTrue((idx == res_idx).all())
        self.assertTrue((sidx == res_sidx).all())
        self.assertTrue((sseg == res_sseg).all())        

    def test_unique_gpu_1d(self):
        with xdl.device("GPU"):
            res_uniq = np.array([0,1,2,3])
            res_idx = np.array([3,1,2,1,3,1,2,0,1,3,1,1,
                                2,0,1,3,3,2,3,2,2,1,2,0])
            res_sidx = np.array([2,6,10, 0,1,2,3,4,5,6,9, 0,2,6,7,8,9,10, 0,1,3,6,7,8])
            res_sseg = np.array([3,11,18,24])
            segment = np.array([3,5,8,10,11,12,16,18,20,22,24], np.int32)
            uniq, idx, sidx, sseg = xdl.unique(data, segment=segment, itype=DataType.int32)
            uniq, idx, sidx, sseg = xdl.execute([uniq, idx, sidx, sseg])
            self.assertTrue((uniq == res_uniq).all())
            self.assertTrue((idx == res_idx).all())
            self.assertTrue((sidx == res_sidx).all())
            self.assertTrue((sseg == res_sseg).all())

    def test_unique_cpu_2d(self):
        data = np.array([3,1,2,1,3,1,2,0,1,3,1,1,2,0,1,3,3,2,3,2,2,1,2,0])
        res_uniq = np.array([[3,2],[1,1],[1,3],[2,0],[2,1],[3,1]])
        res_idx = np.array([5,4,5,3,2,1,3,2,0,0,4,3])
        res_sidx = np.array([4,5,3,2,4,2,4,5,0,5,0,2])
        res_sseg = np.array([2,3,5,8,10,12])
                            
        segment = np.array([2,2,5,6,9,12], np.int32)
        uniq, idx, sidx, sseg = xdl.unique(data.reshape((data.size/2, 2)), segment, itype=DataType.int32)
        uniq, idx, sidx, sseg = xdl.execute([uniq, idx, sidx, sseg])

        self.assertTrue((uniq == res_uniq).all())
        self.assertTrue((idx == res_idx).all())
        self.assertTrue((sidx == res_sidx).all())
        self.assertTrue((sseg == res_sseg).all())

    def test_unique_gpu_2d(self):
        with xdl.device("GPU"):
            res_uniq = np.array([[1,1],[1,3],[2,0],[2,1],[3,1],[3,2]])
            res_idx = np.array([4,3,4,2,1,0,2,1,5,5,3,2])
            segment = np.array([2,2,5,6,9,12], np.int32)
            res_sidx = np.array([3, 2,4, 2,4,5, 0,5, 0,2, 4,5])
            res_sseg = np.array([1,3,6,8,10,12])
            uniq, idx, sidx, sseg = xdl.unique(data.reshape((data.size/2, 2)), segment, itype=DataType.int32)
            uniq, idx, sidx, sseg = xdl.execute([uniq, idx, sidx, sseg])
            self.assertTrue((uniq == res_uniq).all())
            self.assertTrue((idx == res_idx).all())
            self.assertTrue((sidx == res_sidx).all())
            self.assertTrue((sseg == res_sseg).all())

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestUnique)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
