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

idx = np.array([0,1,2,3,2,1,4,5,0,2],dtype=np.int32)
values = np.array([1,1,1,1,1,1,1,1,1,1],dtype=np.float)
segs = np.array([3,6,10],dtype=np.int32)
grps = np.array([2,3,4,6,7,10],dtype=np.int32)
embeds = np.array([[0.1],[0.2],[0.3],[0.4],[0.5],[0.6]],dtype=np.float)
with xdl.device('CPU'):
  embeds_shape = xdl.shape_op(embeds)
grads = np.array([[1],[2],[3]], dtype=np.float)
merged_grads = np.array([[1,2],[3,4],[5,6]],dtype=np.float)
sidx = np.array([0,5,0,3,1,3,5,2,4,5],dtype=np.int32)
sseg = np.array([2,4,7,8,9,10],dtype=np.int32)

sidx_nogrp = np.array([0,2,0,1,0,1,2,1,2,2],dtype=np.int32)

class TestKsumGrad(unittest.TestCase):
    #old_fashion_way
    def test_cpu_with_values(self):
      grps = np.array([],dtype=np.int32)
      ksum_grad = xdl.ksum_grad(embeds_shape, idx, values, segs, grps, sidx_nogrp, sseg, grads)
      ksum_grad = xdl.execute(ksum_grad)
      res = np.array([[4],[3],[6],[2],[3],[3]],dtype=np.float)
      self.assertTrue(np.allclose(ksum_grad, res))

      ksum_grad = xdl.ksum_grad(embeds_shape, idx, values, segs, grps, sidx_nogrp, sseg, grads, average=True)
      ksum_grad = xdl.execute(ksum_grad)
      res = np.array([[1.0833333],[1],[1.75],[0.66666666],[0.75],[0.75]],dtype=np.float)
      self.assertTrue(np.allclose(ksum_grad, res))      

    #new_way      
    def test_cpu_no_values(self):
      grps = np.array([],dtype=np.int32)      
      values = np.array([],dtype=np.float)
      ksum_grad = xdl.ksum_grad(embeds_shape, idx, values, segs, grps, sidx_nogrp, sseg, grads)
      ksum_grad = xdl.execute(ksum_grad)
      res = np.array([[4],[3],[6],[2],[3],[3]],dtype=np.float)
      self.assertTrue(np.allclose(ksum_grad, res))

      ksum_grad = xdl.ksum_grad(embeds_shape, idx, values, segs, grps, sidx_nogrp, sseg, grads, average=True)
      ksum_grad = xdl.execute(ksum_grad)
      res = np.array([[1.0833333],[1],[1.75],[0.66666666],[0.75],[0.75]],dtype=np.float)
      self.assertTrue(np.allclose(ksum_grad, res))
      
    def test_gpu(self):
        with xdl.device("GPU"):
            grps = np.array([],dtype=np.int32)
            ksum_grad = xdl.ksum_grad(embeds_shape, idx, values, segs, grps, sidx_nogrp, sseg, grads)
            ksum_grad = xdl.execute(ksum_grad)
            res = np.array([[4],[3],[6],[2],[3],[3]],dtype=np.float)
            self.assertTrue(np.allclose(ksum_grad, res))

    def test_merged_cpu_with_values(self):
        ksum_grad = xdl.ksum_grad(embeds_shape, idx, values, segs, grps, sidx, sseg, merged_grads)
        ksum_grad = xdl.execute(ksum_grad)
        res = np.array([[7],[5],[12],[3],[5],[6]],dtype=np.float)
        self.assertTrue(np.allclose(ksum_grad, res))

        ksum_grad = xdl.ksum_grad(embeds_shape, idx, values, segs, grps, sidx, sseg, merged_grads, average=True)
        ksum_grad = xdl.execute(ksum_grad)
        res = np.array([[2.5],[2.5],[6],[3],[5],[2]],dtype=np.float)
        self.assertTrue(np.allclose(ksum_grad, res))        

    def test_merged_cpu_no_values(self):
        values = np.array([],dtype=np.float)      
        ksum_grad = xdl.ksum_grad(embeds_shape, idx, values, segs, grps, sidx, sseg, merged_grads)
        ksum_grad = xdl.execute(ksum_grad)
        res = np.array([[7],[5],[12],[3],[5],[6]],dtype=np.float)
        self.assertTrue(np.allclose(ksum_grad, res))

        ksum_grad = xdl.ksum_grad(embeds_shape, idx, values, segs, grps, sidx, sseg, merged_grads, average=True)
        ksum_grad = xdl.execute(ksum_grad)
        res = np.array([[2.5],[2.5],[6],[3],[5],[2]],dtype=np.float)
        self.assertTrue(np.allclose(ksum_grad, res))        

    def test_merged_gpu(self):
        with xdl.device("GPU"):
            ksum_grad = xdl.ksum_grad(embeds_shape, idx, values, segs, grps, sidx_nogrp, sseg, merged_grads)
            ksum_grad = xdl.execute(ksum_grad)
            res = np.array([[7],[5],[12],[3],[5],[6]],dtype=np.float)
            self.assertTrue(np.allclose(ksum_grad, res))

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestKsumGrad)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
