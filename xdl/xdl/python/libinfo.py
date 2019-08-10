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

import os

include_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'include')

lib_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'pybind')

bin_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'bin')

cflags = '-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -I%(include_path)s -L%(lib_path)s -I%(include_path)s/third_party/eigen -lxdl_python_pybind' % {'include_path': include_path, 'lib_path': lib_path}

__version__ = 1.2
