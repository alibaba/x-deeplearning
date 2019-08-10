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
values = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],dtype=np.float)
segs = np.array([3,6,10],dtype=np.int32)
grps = np.array([2,3,4,6,7,10],dtype=np.int32)
embeds = np.array([[0.1],[0.2],[0.3],[0.4],[0.5],[0.6]],dtype=np.float)
sidx = np.array([],dtype=np.int32)
sseg = np.array([],dtype=np.int32)        

class TestKsum(unittest.TestCase):
    def test_cpu_ksum(self):
        grps = np.array([],dtype=np.int32)
        ksum = xdl.ksum(embeds, idx, values, segs, grps, sidx, sseg)
        ksum = xdl.execute(ksum)
        res = np.array([[0.06],[0.09],[0.15]], dtype=np.float)
        self.assertTrue(np.allclose(ksum, res))

    def test_gpu_ksum(self):
        with xdl.device("GPU"):
            grps = np.array([],dtype=np.int32)
            ksum = xdl.ksum(embeds, idx, values, segs, grps, sidx, sseg)
            ksum = xdl.execute(ksum)
            res = np.array([[0.06],[0.09],[0.15]], dtype=np.float)
            self.assertTrue(np.allclose(ksum, res))
        
    def test_cpu_kavg(self):
        grps = np.array([],dtype=np.int32)
        ksum = xdl.ksum(embeds, idx, values, segs, grps, sidx, sseg, average=True)
        ksum = xdl.execute(ksum)
        res = np.array([[0.02],[0.03],[0.0375]], dtype=np.float)
        self.assertTrue(np.allclose(ksum, res))

    def test_gpu_kavg(self):
        with xdl.device("GPU"):
            grps = np.array([],dtype=np.int32)
            ksum = xdl.ksum(embeds, idx, values, segs, grps, sidx, sseg, average=True)
            ksum = xdl.execute(ksum)
            res = np.array([[0.02],[0.03],[0.0375]], dtype=np.float)
            self.assertTrue(np.allclose(ksum, res))

    def test_cpu_merged_ksum(self):
        ksum = xdl.ksum(embeds, idx, values, segs, grps, sidx, sseg)
        ksum = xdl.execute(ksum)
        res = np.array([[0.03, 0.03],[0.04,0.05],[0.05,0.1]],
                       dtype=np.float)
        self.assertTrue(np.allclose(ksum, res))

    def test_gpu_merged_ksum(self):
        with xdl.device("GPU"):
            ksum = xdl.ksum(embeds, idx, values, segs, grps, sidx, sseg)
            ksum = xdl.execute(ksum)
            res = np.array([[0.03, 0.03],[0.04,0.05],[0.05,0.1]],
                           dtype=np.float)
            self.assertTrue(np.allclose(ksum, res))

    def test_cpu_merged_kavg(self):
        ksum = xdl.ksum(embeds, idx, values, segs, grps, sidx, sseg, average=True)
        ksum = xdl.execute(ksum)
        res = np.array([[0.015,0.03],[0.04,0.025],[0.05,0.03333333]],
                       dtype=np.float)
        self.assertTrue(np.allclose(ksum, res))

    def test_gpu_merged_kavg(self):
        with xdl.device("GPU"):
            ksum = xdl.ksum(embeds, idx, values, segs, grps, sidx, sseg, average=True)
            ksum = xdl.execute(ksum)
            res = np.array([[0.015,0.03],[0.04,0.025],[0.05,0.03333333]],
                           dtype=np.float)
            self.assertTrue(np.allclose(ksum, res))

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestKsum)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
