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
import numpy as np
from xdl.python.lib.graph import execute

def _string_to_int8(src):
    return np.array([ord(ch) for ch in src], dtype=np.int8)

def convert_ps_variable(var_list, ckpt_dir, output_dir):
    var_str_list = ','.join(var_list)
    op = xdl.ps_convert_ckpt_variable_op(
        variables = _string_to_int8(var_str_list),
        checkpoint_dir=_string_to_int8(ckpt_dir),
        output_dir=_string_to_int8(output_dir))
    execute(op)
