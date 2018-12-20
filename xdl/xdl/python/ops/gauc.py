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
from xdl.python.ops.init_ops import Zeros
from xdl.python.ops.py_func import dtype_xdl_2_np

def _create_variable(name, dtype):
    from xdl.python.framework.variable import Variable
    return Variable(name, shape=[],
                    dtype=dtype,
                    initializer=Zeros(),
                    trainable=False)

def _update_var(var, delta):
    return xdl.ps_assign_add_op(
        var_name=var.name,
        var_type=var.vtype,
        delta=delta)

def gauc(predicts, labels, indicator, **kwargs):
    namescope = "gauc"
    if "namescope" in kwargs:
        namescope = kwargs["namescope"]

    label_filter = np.array([], dtype=dtype_xdl_2_np(labels.dtype))
    if "filter" in kwargs:
        label_filter = kwargs["filter"]

    auc    = _create_variable(namescope + "/auc", xdl.DataType.double)
    pv_num = _create_variable(namescope + "/pv_num", xdl.DataType.int64)

    cur_auc, cur_pv_num = xdl.gauc_calc_op(labels, predicts, indicator,
            label_filter)

    update_auc    = _update_var(auc, cur_auc)
    update_pv_num = _update_var(pv_num, cur_pv_num)

    with xdl.control_dependencies([update_auc, update_pv_num]):
        auc_value = xdl.ps_pull_op(var_name=namescope + '/auc',
                                   var_type='index',
                                   dtype=xdl.DataType.double)
        pv_num_value = xdl.ps_pull_op(var_name=namescope + '/pv_num',
                                      var_type='index',
                                      dtype=xdl.DataType.int64)
        gauc = xdl.gauc_op(auc_value, pv_num_value)
    return gauc

def batch_gauc(predicts, labels, indicator, **kwargs):
    auc, pv_num = xdl.gauc_calc_op(labels, predicts, indicator)
    return xdl.gauc_op(auc, pv_num)
