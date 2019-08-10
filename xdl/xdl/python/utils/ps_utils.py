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

import os
import ctypes
from xdl.python.model_server.model_server_adapter import ModelServerAdapter
from xdl.python.lib.error import PsError
import time

XDL_LIB_NAME = "libxdl_python_pybind.so"
_LIB_NAME = os.path.dirname(os.path.realpath(
    __file__)) + "/../pybind/" + XDL_LIB_NAME
LIB = ctypes.CDLL(_LIB_NAME)


def connect_to_client(client_addr, ckpt_path, retry_cnt=6):
    i = 0
    while not LIB.TFPS_CONNECT_TO_CLIENT(str(client_addr), str(ckpt_path)) and i < retry_cnt:
        i = i + 1
        time.sleep(30)
    if i == retry_cnt:
        raise PsError("Cannot Connect to Client")


def restart_client():
    if not LIB.TFPS_CONNECTED():
        return True
    if not LIB.TFPS_RESTART_CLIENT():
        return False
    return True


_plugins = []


def add_plugin(plugin):
    global _plugins
    _plugins.append(os.path.abspath(plugin))


def run_ps_cmd(**kwargs):
    cmd = [os.path.dirname(os.path.realpath(__file__)) + "/../../bin/ps"]
    for k, v in kwargs.items():
        if v is None:
            continue
        cmd.append("-" + k)
        cmd.append('"' + str(v) + '"')
    for plugin in _plugins:
        cmd.append("--plugin")
        cmd.append('"' + str(plugin) + '"')
    cmd = " ".join(cmd)
    print "cmd: " + cmd
    ret = os.system(cmd)
    if ret != 0:
        raise ValueError("Run cmd Error, cmd=[%s] exit_code=[%s]" % (cmd, ret))


def run_ps_server(scheduler_kv_path, server_id,
                  sm_dense, sm_sparse, sm_hash, bind_cores):
    return run_ps_cmd(r="server",
                      sp=scheduler_kv_path,
                      si=server_id,
                      smdense=sm_dense,
                      smsparse=sm_sparse,
                      smhash=sm_hash,
                      bc=bind_cores)


def run_ps_scheduler(scheduler_kv_path, server_num,
                     checkpoint_path, smem, snet,
                     sqps, sm_dense, sm_sparse,
                     sm_hash, bind_cores):
    return run_ps_cmd(r="scheduler",
                      sp=scheduler_kv_path,
                      sn=server_num,
                      cp=checkpoint_path,
                      smem=smem,
                      snet=snet,
                      sqps=sqps,
                      smdense=sm_dense,
                      smsparse=sm_sparse,
                      smhash=sm_hash,
                      bc=bind_cores)
