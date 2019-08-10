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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xdl
from xdl.python.framework.gradient import def_gradient
from xdl.python.framework.gradient import SparseGrad

""" python wrapper for gradient op used in sparse_engine."""

@def_gradient("KSum")
def KSumGrad(op, grad):
    with xdl.device('CPU'):
        shape = xdl.shape_op(op.inputs[0])
    return [xdl.ksum_grad(shape,
                          op.inputs[1],
                          op.inputs[2],
                          op.inputs[3],
                          op.inputs[4],
                          op.inputs[5],
                          op.inputs[6],
                          grad[0],
                          average=op.attrs['average'])]

@def_gradient("Tile")
def TileGrad(op, grad):
    return [xdl.tile_grad(op.inputs[0],
                          op.inputs[1],
                          op.inputs[2],
                          op.inputs[3],
                          op.inputs[4],
                          grad[0],
                          length=op.attrs['length'],
                          reverse=op.attrs['reverse'])]

@def_gradient("FeatureMaskOp")
def FeatureMaskOpGrad(op, grad):
    return [xdl.feature_mask_op(
            grad[0],
            op.inputs[1],
            mask_conf=op.attrs['mask_conf'],
            index=op.attrs['index'])]

@def_gradient("IdentityOp")
def IdentityOpGrad(op, grad):
    return [xdl.identity_op(grad[0])]

@def_gradient("PsSparsePullOp")
def PsSparsePullOpGrad(op, grad):
    indices = op.inputs[0]
    return [SparseGrad(grad[0], indices)]

@def_gradient("TakeOp")
def TakeGrad(op, grad):
    with xdl.device('CPU'):
        shape = xdl.shape_op(op.inputs[0])
    g = xdl.take_grad(grad[0], op.inputs[1], shape)
    return [g]
