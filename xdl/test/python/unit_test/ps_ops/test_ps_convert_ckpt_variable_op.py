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
import filecmp
import sys, os

def cmp_file(file1_str, file2_str):
    with open(file1_str, 'r') as file1:
        content1 = file1.read().splitlines()
    with open(file2_str, 'r') as file2:
        content2 = file2.read().splitlines()
    return content1 == content2

class TestPsSaveAndRestoreOp(unittest.TestCase):
    def test_all(self):
        pwd = sys.path[0]
        variables=['emb2', 'dense/kernel', 'dense/bias']
        with xdl.model_scope('train'):
            xdl.convert_ps_variable([x for x in variables], pwd + "/ckpt/ckpt_in", pwd + "/ckpt/ckpt_out", True)
        self.assertTrue(cmp_file(pwd + "/ckpt/ckpt_standard/emb2", pwd + "/ckpt/ckpt_out/emb2"))
        self.assertTrue(cmp_file(pwd + "/ckpt/ckpt_standard/emb2&accumulation", pwd + "/ckpt/ckpt_out/emb2&accumulation"))
        self.assertTrue(cmp_file(pwd + "/ckpt/ckpt_standard/dense&kernel", pwd + "/ckpt/ckpt_out/dense&kernel"))
        self.assertTrue(cmp_file(pwd + "/ckpt/ckpt_standard/dense&kernel&accumulation", pwd + "/ckpt/ckpt_out/dense&kernel&accumulation"))
        self.assertTrue(cmp_file(pwd + "/ckpt/ckpt_standard/dense&bias", pwd + "/ckpt/ckpt_out/dense&bias"))
        self.assertTrue(cmp_file(pwd + "/ckpt/ckpt_standard/dense&bias&accumulation", pwd + "/ckpt/ckpt_out/dense&bias&accumulation"))
        os.system("rm -r " + pwd + "/ckpt/ckpt_out")

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestPsSaveAndRestoreOp)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

