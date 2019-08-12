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
from xdl.python.lib.graph import execute

def _string_to_int8(src):
    return np.array([ord(ch) for ch in src], dtype=np.int8)

class TestPsStreamingTriggerDenseOp(unittest.TestCase):
    def test_all(self):
        op = xdl.ps_streaming_trigger_dense_op(_string_to_int8("inc-1234"))
        execute(op)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestPsStreamingTriggerDenseOp)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

