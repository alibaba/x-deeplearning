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

idx = np.array([2,1,3,0,1,4,0,3,5,2], dtype=np.int32)
values = np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.float)
segs = np.array([3,4,6,6,10],dtype=np.int32)
grps = np.array([],dtype=np.int32)
embeds = np.array([[0.1],[0.2],[0.3],[0.4],[0.5],[0.6]],dtype=np.float)
grads = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9],
                 [1.0,1.1,1.2],[1.3,1.4,1.5]],dtype=np.float)
length = 3

class TestTileGrad(unittest.TestCase):
    def test_cpu_tile_empty_value(self):
        empty_values = np.array([], dtype=np.float)
        res = xdl.tile_grad(embeds, idx, empty_values, segs, grps,
                            grads, length=length, reverse=False)
        res = xdl.execute(res)
        res_grad = np.array([[1.7],[0.9],[0.1],[1.7],[0.8],[1.5]],dtype=np.float)
        self.assertTrue(np.allclose(res, res_grad))

    def test_cpu_tile_empty_value_reverse(self):
        empty_values = np.array([], dtype=np.float)
        res = xdl.tile_grad(embeds, idx, empty_values, segs, grps,
                            grads, length=length, reverse=True)
        res = xdl.execute(res)
        res_grad = np.array([[0.4],[1.0],[1.6],[1.6],[0.7],[1.4]],dtype=np.float)
        self.assertTrue(np.allclose(res, res_grad))

    def test_cpu_tile(self):
        res = xdl.tile_grad(embeds, idx, values, segs, grps,
                            grads, length=length, reverse=False)
        res = xdl.execute(res)
        res_grad = np.array([[10.7],[3.9],[0.1],[12.1],[4.8],[13.5]],dtype=np.float)
        self.assertTrue(np.allclose(res, res_grad))

    def test_cpu_tile_reverse(self):
        res = xdl.tile_grad(embeds, idx, values, segs, grps,
                            grads, length=length, reverse=True)
        res = xdl.execute(res)
        res_grad = np.array([[1.6],[4.4],[13.3],[12.3],[4.2],[12.6]],dtype=np.float)
        self.assertTrue(np.allclose(res, res_grad))

    def test_gpu_tile_empty_value(self):
        with xdl.device("GPU"):
            empty_values = np.array([], dtype=np.float)
            res = xdl.tile_grad(embeds, idx, empty_values, segs, grps,
                                grads, length=length, reverse=False)
            res = xdl.execute(res)
            res_grad = np.array([[1.7],[0.9],[0.1],[1.7],[0.8],[1.5]],dtype=np.float)
            self.assertTrue(np.allclose(res, res_grad))

    def test_gpu_tile_empty_value_reverse(self):
        with xdl.device("GPU"):
            empty_values = np.array([], dtype=np.float)
            res = xdl.tile_grad(embeds, idx, empty_values, segs, grps,
                                grads, length=length, reverse=True)
            res = xdl.execute(res)
            res_grad = np.array([[0.4],[1.0],[1.6],[1.6],[0.7],[1.4]],dtype=np.float)
            self.assertTrue(np.allclose(res, res_grad))

    def test_gpu_tile(self):
        with xdl.device("GPU"):
            res = xdl.tile_grad(embeds, idx, values, segs, grps,
                                grads, length=length, reverse=False)
            res = xdl.execute(res)
            res_grad = np.array([[10.7],[3.9],[0.1],[12.1],[4.8],[13.5]],dtype=np.float)
            self.assertTrue(np.allclose(res, res_grad))

    def test_gpu_tile_reverse(self):
        with xdl.device("GPU"):
            res = xdl.tile_grad(embeds, idx, values, segs, grps,
                                grads, length=length, reverse=True)
            res = xdl.execute(res)
            res_grad = np.array([[1.6],[4.4],[13.3],[12.3],[4.2],[12.6]],dtype=np.float)
            self.assertTrue(np.allclose(res, res_grad))

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestTileGrad)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

