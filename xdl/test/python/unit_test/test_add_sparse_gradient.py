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


class TestAddSparseGradient(unittest.TestCase):
    def test_add_1d(self):
        ids1 = np.array([1,2,3,4,5,6],dtype=np.int64)
        eb1 = np.array([0.1,0.2,0.3,0.4,0.5,0.6],dtype=np.float).reshape((-1,1))
        ids2 = np.array([0,2,3,5,7,9],dtype=np.int64)
        eb2 = np.array([0.,0.2,0.3,0.5,0.7,0.9],dtype=np.float).reshape((-1,1))
        ids3 = np.array([1,4,5,7,8],dtype=np.int64)
        eb3 = np.array([0.1,0.4,0.5,0.7,0.8],dtype=np.float).reshape((-1,1))
        res_eb = np.array([0.2,0.4,0.6,0.8,1.5,0.6,0.0,1.4,0.9,0.8],dtype=np.float).reshape((-1,1))
        res_id = np.array([1,2,3,4,5,6,0,7,9,8],dtype=np.int64)
        add_sparse = xdl.sparse_grad_add_op([eb1,eb2,eb3],[ids1,ids2,ids3])
        eb, id = xdl.execute(add_sparse)
        self.assertTrue((res_id == id).all())
        self.assertTrue(np.allclose(res_eb, eb))

    def test_add_1d_gpu(self):
        ids1 = np.array([1,2,3,4,5,6],dtype=np.int64)
        eb1 = np.array([0.1,0.2,0.3,0.4,0.5,0.6],dtype=np.float).reshape((-1,1))
        ids2 = np.array([0,2,3,5,7,9],dtype=np.int64)
        eb2 = np.array([0.,0.2,0.3,0.5,0.7,0.9],dtype=np.float).reshape((-1,1))
        ids3 = np.array([1,4,5,7,8],dtype=np.int64)
        eb3 = np.array([0.1,0.4,0.5,0.7,0.8],dtype=np.float).reshape((-1,1))
        res_eb = np.array([0.2,0.4,0.6,0.8,1.5,0.6,0.0,1.4,0.9,0.8],dtype=np.float).reshape((-1,1))
        res_id = np.array([1,2,3,4,5,6,0,7,9,8],dtype=np.int64)
        with xdl.device("GPU"):
            add_sparse = xdl.sparse_grad_add_op([eb1,eb2,eb3],[ids1,ids2,ids3])
            eb, id = xdl.execute(add_sparse)
            self.assertTrue((res_id == id).all())
            self.assertTrue(np.allclose(res_eb, eb))

    def test_add_2d(self):
        ids1 = np.array([0,1,2,3,4,5],dtype=np.int64).reshape((-1,2))
        eb1 = np.array([0.,0.1,0.2,0.3,0.4,0.6],dtype=np.float).reshape((-1,2))
        ids2 = np.array([0,1,4,5,6,7],dtype=np.int64).reshape((-1,2))
        eb2 = np.array([0.,0.1,0.4,0.5,0.6,0.7],dtype=np.float).reshape((-1,2))
        ids3 = np.array([2,3,4,6],dtype=np.int64).reshape((-1,2))
        eb3 = np.array([0.2,0.3,0.4,0.6],dtype=np.float).reshape((-1,2))
        res_id = np.array([0,1,2,3,4,5,6,7,4,6],dtype=np.int64).reshape((-1,2))
        res_eb = np.array([0.,0.2,0.4,0.6,0.8,1.1,0.6,0.7,0.4,0.6],dtype=np.float).reshape((-1,2))
        add_sparse = xdl.sparse_grad_add_op([eb1,eb2,eb3],[ids1,ids2,ids3])
        eb, id = xdl.execute(add_sparse)
        self.assertTrue((res_id == id).all())
        self.assertTrue(np.equal(res_id, id).all())
        self.assertTrue(np.allclose(res_eb, eb))

    def test_add_2d_gpu(self):
        ids1 = np.array([0,1,2,3,4,5],dtype=np.int64).reshape((-1,2))
        eb1 = np.array([0.,0.1,0.2,0.3,0.4,0.6],dtype=np.float).reshape((-1,2))
        ids2 = np.array([0,1,4,5,6,7],dtype=np.int64).reshape((-1,2))
        eb2 = np.array([0.,0.1,0.4,0.5,0.6,0.7],dtype=np.float).reshape((-1,2))
        ids3 = np.array([2,3,4,6],dtype=np.int64).reshape((-1,2))
        eb3 = np.array([0.2,0.3,0.4,0.6],dtype=np.float).reshape((-1,2))
        res_id = np.array([0,1,2,3,4,5,6,7,4,6],dtype=np.int64).reshape((-1,2))
        res_eb = np.array([0.,0.2,0.4,0.6,0.8,1.1,0.6,0.7,0.4,0.6],dtype=np.float).reshape((-1,2))
        with xdl.device("GPU"):
            add_sparse = xdl.sparse_grad_add_op([eb1,eb2,eb3],[ids1,ids2,ids3])
            eb, id = xdl.execute(add_sparse)
            self.assertTrue(np.equal(res_id, id).all())
            self.assertTrue(np.allclose(res_eb, eb))

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestAddSparseGradient)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
