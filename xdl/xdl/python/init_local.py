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

from xdl.python.pybind import DataType, RunOption, RunStatistic
from xdl.python.lib.internal_ops import *
from xdl.python.lib.datatype import *
from xdl.python.libinfo import include_path, lib_path, bin_path
from xdl.python.lib.error import ArgumentError, IndexOverflow, PsError, InternalError, OutOfRange
from xdl.python.lib.op_generate import load_op_library
from xdl.python.lib.graph import Graph, execute, execute_loop, execute_loop_wait, namescope, device, control_dependencies
from xdl.python.lib.tensor import Tensor, convert_to_tensor
from xdl.python.ops.py_func import py_func
from xdl.python.pybind import fs
from xdl.python.pybind import parsers
from xdl.python.pybind import features
from xdl.python.pybind import GetIOP
from xdl.python.sparse_engine.base import SparseTensor, MergedSparseTensor
from xdl.python.io.data_io import DataIO
from xdl.python.io.data_reader import DataReader
from xdl.python.io.data_sharding import DataSharding, SwiftSharding
from xdl.python.framework.session import Session, Hook

'''
from xdl.python.ops.ps_ops import *
from xdl.python.framework.gradient import def_gradient, def_gradient_internal, gradient, get_sparse_grads
from xdl.python.framework.variable import Variable, trainable_variables, global_variables, global_initializers, variable_registers, get_variable_by_name
from xdl.python.sparse_engine.embedding import embedding, merged_embedding
from xdl.python.training.gradient_utils import get_gradient, get_gradients
from xdl.python.ops.init_ops import *
from xdl.python.utils.config import *
from xdl.python.utils.timeline import Timeline
from xdl.python.utils.collections import *
from xdl.python.backend.model_scope import model_scope
from xdl.python.ops.auc import auc, batch_auc
from xdl.python.ops.gauc import gauc, batch_gauc
from xdl.python import preload as _preload
from xdl.python.training.optimizer_impls import SGD, Momentum, Adagrad, Adam, Ftrl
from xdl.python.training.gradient_utils import get_gradients, get_gradient
from xdl.python.training.train_session import TrainSession, LoggerHook, SyncRunHook, SemiSyncRunHook, BarrierHook
from xdl.python.training.estimator import Estimator
from xdl.python.training.env import current_env
import xdl.python.ops.ops_grad
from xdl.python.training.saver import graph_tag, Saver, CheckpointHook
from xdl.python.model_server.model_server import ModelServer
'''

