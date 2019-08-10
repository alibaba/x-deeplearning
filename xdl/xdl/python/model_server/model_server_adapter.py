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

from xdl.python import pybind
from xdl.python.lib.error import check_error
from xdl.python.lib.internal_ops import ps_model_server_forward_wait_op
from xdl.python.lib.internal_ops import ps_model_server_forward_request_op
from xdl.python.lib.internal_ops import ps_model_server_forward_response_op
from xdl.python.lib.internal_ops import ps_model_server_backward_wait_op
from xdl.python.lib.internal_ops import ps_model_server_backward_request_op
from xdl.python.lib.internal_ops import ps_model_server_backward_response_op
from xdl.python.lib.graph import control_dependencies

class ModelServerAdapter(object):
  def __init__(self, scheduler, server_type, server_id, forward_spec, backward_spec, dtype):
    self.service = pybind.ModelServer(scheduler, server_type, server_id, forward_spec, backward_spec)

    self._forward_wait = ps_model_server_forward_wait_op(handle=self.service.forward_handle())
    self._forward_handle, self._forward_ids = ps_model_server_forward_request_op(handle=self.service.forward_handle())
    self._backward_wait = ps_model_server_backward_wait_op(handle=self.service.backward_handle())
    self._backward_handle, self._backward_ids, self._backward_grads = ps_model_server_backward_request_op(handle=self.service.backward_handle(), dtype=dtype)

  def init(self):
    check_error(self.service.init())

  def forward_wait(self):
    return self._forward_wait

  def forward_ids(self):
    return self._forward_ids

  def forward_result(self, result):
    return ps_model_server_forward_response_op(self._forward_handle, result)

  def backward_wait(self):
    return self._backward_wait

  def backward_ids(self):
    return self._backward_ids

  def backward_grads(self):
    return self._backward_grads

  def backward_result(self, result):
    with control_dependencies(result):
      return ps_model_server_backward_response_op(self._backward_handle)

