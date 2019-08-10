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
from xdl.python.framework.variable import Variable

_GLOBAL_STEP = None

def get_global_step():
    global _GLOBAL_STEP
    if _GLOBAL_STEP is not None:
        return _GLOBAL_STEP
    _GLOBAL_STEP = Variable(
        name = "xdl_global_step", 
        shape = [],
        dtype = xdl.DataType.int64,
        trainable = False,
        initializer = xdl.Zeros())
    return _GLOBAL_STEP

