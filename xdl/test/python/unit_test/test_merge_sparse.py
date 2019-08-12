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

ids1 = np.array([1,2,3,4,5],dtype=np.int64)
ids2 = np.array([6,7,8,9,10],dtype=np.int64)
val1 = np.array([0,0,0,0,0],dtype=np.float32)
val2 = np.array([1,1,1,1,1],dtype=np.float32)
seg1 = np.array([2,3,5],dtype=np.int32)
seg2 = np.array([1,4,5],dtype=np.int32)

ids3 = np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.int64).reshape((5,2))
ids4 = np.array([11,12,13,14,15,16,17,18,19,20],dtype=np.int64).reshape((5,2))

class TestMergeSparse(unittest.TestCase):
    def test_cpu(self):
        merged_sparse = xdl.merge_sparse_op([ids1,ids2],[val1,val2],[seg1,seg2])
        ids,vals,segs,grps = xdl.execute(merged_sparse)
        res_ids = np.array([1,2,6,3,7,8,9,4,5,10],dtype=np.int64)
        res_vals = np.array([0,0,1,0,1,1,1,0,0,1],dtype=np.float)
        res_segs = np.array([3,7,10])
        res_grps = np.array([2,3,4,7,9,10])
        self.assertTrue((ids == res_ids).all())
        self.assertTrue(np.allclose(vals, res_vals))
        self.assertTrue((segs == res_segs).all())
        self.assertTrue((grps == res_grps).all())

    def test_cpu_empty_val(self):
        val1 = np.array([],dtype=np.float32)
        val2 = np.array([],dtype=np.float32)
        merged_sparse = xdl.merge_sparse_op([ids1,ids2],[val1,val2],[seg1,seg2])
        ids,vals,segs,grps = xdl.execute(merged_sparse)
        res_ids = np.array([1,2,6,3,7,8,9,4,5,10],dtype=np.int64)
        res_vals = np.array([],dtype=np.float)
        res_segs = np.array([3,7,10])
        res_grps = np.array([2,3,4,7,9,10])
        self.assertTrue((ids == res_ids).all())
        self.assertTrue(np.allclose(vals, res_vals))
        self.assertTrue((segs == res_segs).all())
        self.assertTrue((grps == res_grps).all())

    def test_cpu_2d(self):
        merged_sparse = xdl.merge_sparse_op([ids3,ids4],[val1,val2],[seg1, seg2])
        ids,vals,segs,grps = xdl.execute(merged_sparse)
        res_ids = np.array([[1,2],[3,4],[11,12],[5,6],[13,14],[15,16],[17,18],[7,8],[9,10],[19,20]])
        res_vals = np.array([0,0,1,0,1,1,1,0,0,1])
        res_segs = np.array([3,7,10])
        res_grps = np.array([2,3,4,7,9,10])
        self.assertTrue((ids == res_ids).all())
        self.assertTrue(np.allclose(vals, res_vals))
        self.assertTrue((segs == res_segs).all())
        self.assertTrue((grps == res_grps).all())

    def test_gpu(self):
        with xdl.device("GPU"):
            merged_sparse = xdl.merge_sparse_op([ids1,ids2],[val1,val2],[seg1,seg2])
            ids,vals,segs,grps = xdl.execute(merged_sparse)
            res_ids = np.array([1,2,6,3,7,8,9,4,5,10],dtype=np.int64)
            res_vals = np.array([0,0,1,0,1,1,1,0,0,1],dtype=np.float)
            res_segs = np.array([3,7,10])
            res_grps = np.array([2,3,4,7,9,10])
            self.assertTrue((ids == res_ids).all())
            self.assertTrue(np.allclose(vals, res_vals))
            self.assertTrue((segs == res_segs).all())
            self.assertTrue((grps == res_grps).all())

    def test_gpu_empty_val(self):
        with xdl.device("GPU"):
            val1 = np.array([],dtype=np.float32)
            val2 = np.array([],dtype=np.float32)
            merged_sparse = xdl.merge_sparse_op([ids1,ids2],[val1,val2],[seg1,seg2])
            ids,vals,segs,grps = xdl.execute(merged_sparse)
            res_ids = np.array([1,2,6,3,7,8,9,4,5,10],dtype=np.int64)
            res_vals = np.array([],dtype=np.float)
            res_segs = np.array([3,7,10])
            res_grps = np.array([2,3,4,7,9,10])
            self.assertTrue((ids == res_ids).all())
            self.assertTrue(np.allclose(vals, res_vals))
            self.assertTrue((segs == res_segs).all())
            self.assertTrue((grps == res_grps).all())

    def test_gpu_2d(self):
        with xdl.device("GPU"):
            merged_sparse = xdl.merge_sparse_op([ids3,ids4],[val1,val2],[seg1, seg2])
            ids,vals,segs,grps = xdl.execute(merged_sparse)
            res_ids = np.array([[1,2],[3,4],[11,12],[5,6],[13,14],[15,16],[17,18],[7,8],[9,10],[19,20]])
            res_vals = np.array([0,0,1,0,1,1,1,0,0,1])
            res_segs = np.array([3,7,10])
            res_grps = np.array([2,3,4,7,9,10])
            self.assertTrue((ids == res_ids).all())
            self.assertTrue(np.allclose(vals, res_vals))
            self.assertTrue((segs == res_segs).all())
            self.assertTrue((grps == res_grps).all())

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestMergeSparse)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
