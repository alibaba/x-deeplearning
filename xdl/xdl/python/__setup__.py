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

#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.install import install

import os

allowed_extensions = [".so", ".h"]

kwargs = {'install_requires': ['protobuf==3.6.1', 'pyhdfs', 'matplotlib==2.0.0'], 'zip_safe': False}

class PostCmd(install):
  def run(self):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    self.spawn(['cp', os.path.join(dir_name, '../third_party/librdkafka/src/librdkafka.so'), '/usr/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_release/libtbbmalloc_proxy.so.2"), '/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_release/libtbbmalloc_proxy.so"), '/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_release/libtbb.so.2"), '/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_release/libtbb.so"), '/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_release/libtbbmalloc.so.2"), '/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_release/libtbbmalloc.so"), '/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_debug/libtbbmalloc_proxy_debug.so.2"), '/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_debug/libtbbmalloc_proxy_debug.so"), '/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_debug/libtbb_debug.so.2"), '/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_debug/libtbb_debug.so"), '/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_debug/libtbbmalloc_debug.so.2"), '/lib/'])
    self.spawn(['cp', os.path.join(dir_name, "../../build/tbb_cmake_build/tbb_cmake_build_subdir_debug/libtbbmalloc_debug.so"), '/lib/'])        
    install.run(self)

def find_package_data():
  ret = []
  for root, dirs, files in os.walk("./xdl"):
    root = root[6:]
    for name in files:
      ext = os.path.splitext(name)[1]
      if ext in allowed_extensions or root.find('third_party/eigen') != -1:
        ret.append(root + '/' + name)
  ret.append("bin/ps")
  ret.append("bin/protoc")
  return ret

setup(name='xdl',
      version='1.2',
      packages=find_packages(where='.'),
      package_data={'xdl':find_package_data()},
      cmdclass={'install': PostCmd},
      **kwargs)
