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

import mxnet as mx
from mxnet.base import _Null
from xdl.python.lib.tensor import Tensor
from xdl.python.utils.collections import *

RawBatchNorm = mx.sym.BatchNorm

def BatchNormWrapper(data=None, gamma=None, 
                     beta=None, moving_mean=None, 
                     moving_var=None, eps=_Null, 
                     momentum=0.9, fix_gamma=_Null, 
                     use_global_stats=_Null, 
                     output_mean_var=_Null, 
                     axis=_Null, 
                     cudnn_off=_Null, 
                     name=None, 
                     attr=None, 
                     out=None, 
                     **kwargs):
  outputs = RawBatchNorm(data=data, gamma=gamma, 
                         beta=beta, 
                         moving_mean=moving_mean, 
                         moving_var=moving_var, 
                         eps=eps, 
                         momentum=momentum, 
                         fix_gamma=fix_gamma, 
                         use_global_stats=use_global_stats, 
                         output_mean_var=True, 
                         axis=axis, cudnn_off=cudnn_off, 
                         name=name, attr=attr, 
                         out=out, **kwargs)
  if eps == _Null:
    eps = 1e-3
  var = (1/outputs[2]) ** 2 - eps
  aux_list = outputs.list_auxiliary_states()
  add_to_collection(MXNET_BN_STATISTIC, zip(
      aux_list[len(aux_list)-2:], [outputs[1], var], [momentum]*2))
  if output_mean_var == _Null:
    return outputs[0]
  else:
    return outputs

mx.sym.BatchNorm = BatchNormWrapper

RawBatchNormV1 = mx.sym.BatchNorm_v1

def BatchNormV1Wrapper(data=None, gamma=None, 
                       beta=None, eps=_Null, 
                       momentum=0.9, fix_gamma=_Null, 
                       use_global_stats=_Null, 
                       output_mean_var=_Null, 
                       name=None, attr=None, 
                       out=None, **kwargs):
  outputs = RawBatchNormV1(data=data, gamma=gamma,
                           beta=beta, eps=eps, 
                           momentum=momentum, 
                           fix_gamma=fix_gamma, 
                           use_global_stats=use_global_stats, 
                           output_mean_var=True, 
                           name=name, attr=attr, 
                           out=out, **kwargs)
  aux_list = outputs.list_auxiliary_states()
  add_to_collection(MXNET_BN_STATISTIC, zip(
      aux_list[len(aux_list)-2:], outputs[1:], [momentum]*2))
  if output_mean_var == _Null:
    return outputs[0]
  else:
    return outputs

mx.sym.BatchNorm_v1 = BatchNormV1Wrapper
  
