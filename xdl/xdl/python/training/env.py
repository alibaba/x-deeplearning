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

import contextlib
import traceback
import time
from xdl.python.utils.config import *
from xdl.python.utils.ps_utils import connect_to_client, restart_client, run_ps_server, run_ps_scheduler
from xdl.python.model_server.model_server import get_model_server_by_name, set_model_server_id
from xdl.python.model_server.model_server_adapter import ModelServerAdapter
from xdl.python.lib.graph import execute, execute_loop_wait
from xdl.python.framework.variable import global_variables, variable_registers, global_initializers
from xdl.python.backend.model_scope import model_scope
from xdl.python.lib.error import PsError

class Env(object):
  def __init__(self):
    self._task_name = None
    self._model_server = []
    self._model_server_num = []

  def _ps_do(self):
    raise Exception("unimplemented method")
  def _scheduler_do(self):
    raise Exception("unimplemented method")
  def _worker_do(self):
    raise Exception("unimplemented method")
  def _model_server_do(self):
    raise Exception("unimplemented method")

  def _ps_sess(self):
    pass
  def _scheduler_sess(self):
    pass
  def _worker_sess(self):
    pass
  def _model_server_sess(self):
    pass

  def _start(self):
    if self._task_name == 'ps':
      self._ps_do()
    elif self._task_name == 'worker':
      self._worker_do()
    elif self._task_name == 'scheduler':
      self._scheduler_do()      
    elif self._task_name in self._model_server:
      self._model_server_do()
    else:
      pass

  def sess_start(self):
    if self._task_name == 'ps':
      self._ps_sess()
    elif self._task_name == 'worker':
      self._worker_sess()
    elif self._task_name == 'scheduler':
      self._scheduler_sess()      
    elif self._task_name in self._model_server:
      self._model_server_sess()
    else:
      pass

  @property
  def is_chief(self):
    if get_task_index() is not None:
      return get_task_index() == 0
    else:
      return True

class LocalEnv(Env):
  def __init__(self):
    super(LocalEnv, self).__init__()
    self._task_name = 'worker'
    self._start()
  def _ps_do(self):
    pass
  def _scheduler_do(self):
    pass
  def _worker_do(self):
    connect_to_client("localhost", get_ckpt_dir())

class DistributedEnv(Env):
  def __init__(self):
    super(DistributedEnv, self).__init__()
    self._zk_addr = get_zk_addr()    
    self._task_name = get_task_name()
    self._task_index = get_task_index()
    self._bind_core = get_config("bind_cores")
    self._model_bank = get_config("checkpoint", "model_bank")
    if self._bind_core is None:
      self._bind_core = False
    self._is_chief = True if self._task_index == 0 else False
    self._sm_dense = None
    self._sm_sparse = None
    self._sm_hash = None
    if self._task_name == 'scheduler':
      if get_config():
        self._ps_num = get_config("ps", "instance_num")
        self._ps_memory_m = get_config("ps", "memory_m")
        self._ps_cpu_cores = get_config("ps", "cpu_cores")
        self._ckpt_dir = get_config("checkpoint", "output_dir")
      else:
        self._ps_num = int(get_ps_num())
        self._ps_memory_m = int(get_ps_memory_m())
        self._ps_cpu_cores = int(get_ps_cpu_cores())
        self._ckpt_dir = get_ckpt_dir()

      if self._ps_num is None:
        raise Exception("ps_num is not specified")
      if self._ps_memory_m is None:
        raise Exception("ps_memory_m is not specified")        
      if self._ps_cpu_cores is None:
        raise Exception("ps_cpu_cores is not specified")        
      if self._ckpt_dir is None:
        raise Exception("ckpt_dir is not specified")        

    if get_config():
      self._model_server = get_config("model_server") or []
      self._model_server_num = [get_config("extend_role", i, "instance_num") for i in self._model_server]
    else:
      self._model_server = get_model_server()
      self._model_server_num = get_model_server_num()
    for i in range(len(self._model_server)):
      set_model_server_id(self._model_server[i], i + 1)
    self._model_server_num_dict = dict(zip(self._model_server, self._model_server_num))

    if self._task_name != "worker":
      config = get_config("streaming_output")
      if config:
        if "dense" in config:
          self._sm_dense = config["dense"]["addr"]
        if "sparse" in config:
          self._sm_sparse = config["sparse"]["addr"]
        if "hash" in config:
          self._sm_hash = config["hash"]["addr"]
    self._start()

  def _ps_do(self):
    if not get_ps_mode():
      return
    run_ps_server(
        scheduler_kv_path = self._zk_addr,
        server_id = self._task_index,
        sm_dense = self._sm_dense,
        sm_sparse = self._sm_sparse,
        sm_hash = self._sm_hash,
        bind_cores = self._bind_core)

  def _scheduler_do(self):
    if not get_ps_mode():
      return
    run_ps_scheduler(
        scheduler_kv_path = self._zk_addr,
        server_num = ','.join([str(self._ps_num)] + [str(i) for i in self._model_server_num]),
        checkpoint_path = self._ckpt_dir,
        smem = int(self._ps_memory_m / 3),
        snet = 4000 * self._ps_cpu_cores / 64,
        sqps = 1000000 * self._ps_cpu_cores / 64,
        sm_dense = self._sm_dense,
        sm_sparse = self._sm_sparse,
        sm_hash = self._sm_hash,
        bind_cores = self._bind_core)

  def _worker_do(self):
    if not get_ps_mode():
      return
    if self._zk_addr:
      connect_to_client(self._zk_addr, '')

  def _model_server_do(self):
    pass

  def _model_server_sess(self):
    self.start_model_server(get_model_server_by_name(self._task_name))

  def start_model_server(self, model_server):
    if model_server.name() != self._task_name:
      return
    model_server = get_model_server_by_name(self._task_name)
    model_server_adapter = ModelServerAdapter(self._zk_addr, self._model_server.index(self._task_name) + 1, self._task_index, model_server.forward_cache, model_server.backward_cache, model_server.dtype())
    model_server_adapter.init()
    model_server.init_server(model_server_adapter)
    with model_scope("ams_gear_forward"):
      while True:
        try:
          connect_to_client(self._zk_addr, '')
          if variable_registers() is not None:
            execute(variable_registers())
            execute(global_initializers())
          break
        except PsError as e:
          traceback.print_exc()
          time.sleep(10)
    model_server.run_server()
    while True:
      try:
        while True:
          print "RESTARTING CLIENT"
          if restart_client():
            break
          time.sleep(10)
        execute_loop_wait()
      except PsError as e:
        traceback.print_exc()
        time.sleep(10)

  def model_server_size(self, name):
    return self._model_server_num_dict[name]

  def task_name(self):
    return self._task_name

  def task_id(self):
    return self._task_index

def init_env():
  if get_run_mode() == 'local':
    return LocalEnv()
  else:
    return DistributedEnv()

_ENV = init_env()

def current_env():
  global _ENV
  return _ENV

def is_local_mode():
  return get_run_mode() == 'local'

