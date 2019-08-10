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

from xdl.python.lib.graph import execute_loop, execute_loop_wait
from xdl.python.lib.error import PsError
import traceback
from xdl.python.lib.internal_ops import ps_model_server_client_forward_op
from xdl.python.lib.internal_ops import ps_model_server_client_backward_op
from xdl.python.training.gradient_utils import get_gear_gradient

_MODEL_DICT = {}
_MODEL_ID = {}

class ModelServer(object):
  class Forward(object):
    class NoCache(object):
      def __str__(self):
        return "name=no_cache"
    class UniqueCache(object):
      def __init__(self, window_size):
        self._window_size = window_size
      def __str__(self):
        return "name=unique_cache&window_size=" + str(self._window_size)
  class Backward(object):
    class NoCache(object):
      def __str__(self):
        return "name=no_cache"
    class UniqueCache(object):
      def __init__(self, window_size):
        self._window_size = window_size
      def __str__(self):
        return "name=unique_cache&window_size=" + str(self._window_size)

  def __init__(
      self, name,
      model,
      dtype,
      forward_cache=None,
      backward_cache=None,
      forward_thread=None,
      backward_thread=None):

    if forward_cache is None: forward_cache = ModelServer.Forward.NoCache()
    if backward_cache is None: backward_cache = ModelServer.Backward.NoCache()
    if forward_thread is None: forward_thread = 10
    if backward_thread is None: backward_thread = 10

    self._name = name
    self._model = model
    self._dtype = dtype
    self.forward_cache = str(forward_cache)
    self.backward_cache = str(backward_cache)
    self.forward_thread = forward_thread
    self.backward_thread = backward_thread

    global _MODEL_DICT
    _MODEL_DICT[name] = self

  def __call__(self, x):
    i = _MODEL_ID[self._name]
    self._ids = x
    self._rst = ps_model_server_client_forward_op(x, i, self._dtype)
    return self._rst

  def update(self):
    i = _MODEL_ID[self._name]
    return ps_model_server_client_backward_op(self._ids, get_gear_gradient(self._rst), i)

  def init_server(self, adapter):
    self._forward, self._backward = self._model(
        adapter.forward_ids(),
        adapter.backward_ids(),
        adapter.backward_grads())
    self._forward_wait = adapter.forward_wait()
    self._backward_wait = adapter.backward_wait()
    self._forward = adapter.forward_result(self._forward)
    self._backward = adapter.backward_result(self._backward)

  def run_server(self):
    for i in range(self.forward_thread):
      execute_loop(self._forward_wait, self._forward)
    for i in range(self.backward_thread):
      execute_loop(self._backward_wait, self._backward)

  def dtype(self):
    return self._dtype

  def name(self):
    return self._name

  @staticmethod
  def updates():
    return [i.update() for i in _MODEL_DICT.values()]

def get_model_server_by_name(name):
  return _MODEL_DICT[name]

def set_model_server_id(name, i):
  global _MODEL_ID
  _MODEL_ID[name] = i
