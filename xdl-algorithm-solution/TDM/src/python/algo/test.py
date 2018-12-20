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
import os
import ctypes
from xdl.python import pybind

import struct

from store import Store
from dist_tree import DistTree

s = Store('')
s.load('mock_tree.pb')

print(repr(s.get(struct.pack('L', 0)[::-1])))

tree = DistTree()
tree.set_store(s.get_handle())
tree.load()

#_LIB_NAME = "/home/huimin.yhm/dist_tree/build_release/dist_tree/libdist_tree.so"
_LIB_NAME = "/home/a/anaconda/lib/python2.7/site-packages/dist_tree-0.1-py2.7.egg/dist_tree/libselector.so"
_LIB_NAME = "/home/pengye.zpy/dist_tree/build/dist_tree/libselector.so"
ctypes.CDLL(_LIB_NAME, ctypes.RTLD_GLOBAL)

top = pybind.GetIOP("TDMOP")
#dop = pybind.GetIOP("DebugOP")

data_io = pybind.DataIO("ttt", 1024, pybind.FSType.local, 3)
print data_io

#debug_op = pybind.DebugOP()

data_io.add_path('xxx')
#data_io.add_op(dop)
data_io.add_op(top)

data_io.startup()
