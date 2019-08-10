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

class Regularizer(object):
    def __call__(self, var):
        raise Exception("unimplement call")        

class L1Loss(Regularizer):
    def __init__(self, l1):
        self._l1 = l1

    def __call__(self, var):
        return xdl.l1_loss(var.value, self._l1)

class L2Loss(Regularizer):
    def __init__(self, l2):
        self._l2 = l2

    def __call__(self, var):
        return xdl.l2_loss(var.value, self._l2)
        
