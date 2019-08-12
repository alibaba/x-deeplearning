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
import numpy as np
from xdl.python.ops.init_ops import Zeros

def _create_auc_variable(name, num_thresholds):
    from xdl.python.framework.variable import Variable
    return Variable(name, shape=[num_thresholds],
                    dtype=xdl.DataType.int64,
                    initializer=Zeros(),
                    trainable=False,
                    save="false")

def _update_auc(var, delta):
    return xdl.ps_assign_add_op(
        var_name=var.name,
        var_type=var.vtype,
        delta=delta)

def reset_auc_variable_op(num_thresholds, name):
    return xdl.ps_assign_op(
        var_name=name,
        var_type="index",
        delta=np.zeros((num_thresholds), dtype=np.int64))

def reset_auc_variables_op(num_thresholds, namescope="auc"):
    ops = []
    ops.append(reset_auc_variable_op(num_thresholds, namescope + "/tp"))
    ops.append(reset_auc_variable_op(num_thresholds, namescope + "/fp"))
    ops.append(reset_auc_variable_op(num_thresholds, namescope + "/tn"))
    ops.append(reset_auc_variable_op(num_thresholds, namescope + "/fn"))
    return ops

def auc(predictions, labels, **kwargs):
    """compute accumulative auc
       Args:
         predictions: prediction tensor
         labels: label tensor
         num_thresholds: the number of thresholds to use when discretizing the roc curve
       Returns:
         auc: a float auc value
    """
    num_thresholds = 200
    namescope = "auc"
    if "num_thresholds" in kwargs:
        num_thresholds = kwargs["num_thresholds"]
    if "namescope" in kwargs:
        namescope = kwargs["namescope"]
    # variable create to score the total tp,fp, tn, fn value
    tp = _create_auc_variable(namescope + "/tp", num_thresholds)
    fp = _create_auc_variable(namescope + "/fp", num_thresholds)
    tn = _create_auc_variable(namescope + "/tn", num_thresholds)
    fn = _create_auc_variable(namescope + "/fn", num_thresholds)

    if predictions is None and labels is None:
      auc = xdl.auc_op(tp.value, fp.value, tn.value, fn.value)
      return auc

    cur_tp, cur_fp, cur_tn, cur_fn = xdl.confusion_matrix_op(
        predictions=predictions,
        labels=labels,
        num_thresholds=num_thresholds)

    update_tp = _update_auc(tp, cur_tp)
    update_fp = _update_auc(fp, cur_fp)
    update_tn = _update_auc(tn, cur_tn)
    update_fn = _update_auc(fn, cur_fn)
    with xdl.control_dependencies([update_tp, update_fp, update_tn, update_fn]):
        auc = xdl.auc_op(tp.value, fp.value, tn.value, fn.value)
    return auc

def batch_auc(predictions, labels, **kwargs):
    """compute auc in a batch
       Args:
         predictions: prediction tensor
         labels: label tensor
         num_thresholds: the number of thresholds to use when discretizing the roc curve
       Returns:
         auc: a float auc value
    """
    num_thresholds = 200
    if "num_thresholds" in kwargs:
        num_thresholds = kwargs["num_thresholds"]
    cur_tp, cur_fp, cur_tn, cur_fn = xdl.confusion_matrix_op(
        predictions=predictions,
        labels=labels,
        num_thresholds=num_thresholds)
    return xdl.auc_op(cur_tp, cur_fp, cur_tn, cur_fn)
