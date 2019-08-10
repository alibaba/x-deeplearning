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

labels = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float)
predicts = np.array([[0.7, 0.3], [0.6, 0.4], [0.1, 0.9], [0.76, 0.24], [0.83,0.17], [0.12, 0.88]], dtype=np.float)
indicator = np.array([0, 1, 1, 1, 2, 2],dtype=np.int32)

class TestGauc(unittest.TestCase):
    def test_gauc(self):
        filter_label = np.array([], dtype=np.float)
        gauc, pv_num = xdl.gauc_calc_op(labels, predicts, indicator,
                filter=filter_label)
        res = xdl.gauc_op(gauc, pv_num)
        gauc, pv_num, res = xdl.execute([gauc, pv_num, res])
        gauc_val = np.array([3.0], dtype=np.float32)
        pv_num_val = np.array([3], dtype=np.int32)
        gauc_res = np.array([1.], dtype=np.float32)
        self.assertTrue(np.allclose(gauc, gauc_val))
        self.assertTrue((pv_num == pv_num_val).all())
        self.assertTrue(np.allclose(res, gauc_res))

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestGauc)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
