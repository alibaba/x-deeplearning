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

from xdl.python.ops.initializer_impls import IdentityInitializer as Identity
from xdl.python.ops.initializer_impls import ConstantInitializer as Constant
from xdl.python.ops.initializer_impls import ZeroInitializer as Zeros
from xdl.python.ops.initializer_impls import OneInitializer as Ones
from xdl.python.ops.initializer_impls import RandomNormalInitializer as RandomNormal
from xdl.python.ops.initializer_impls import RandomUniformInitializer as RandomUniform
from xdl.python.ops.initializer_impls import TruncatedNormalInitializer as TruncatedNormal
from xdl.python.ops.initializer_impls import NormalInitializer as Normal
from xdl.python.ops.initializer_impls import UniformUnitScalingInitializer as UniformUnitScaling
from xdl.python.ops.initializer_impls import VarianceScalingInitializer as VarianceScaling
from xdl.python.ops.initializer_impls import OrthogonalInitializer as Orthogonal



