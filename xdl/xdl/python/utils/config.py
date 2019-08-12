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

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="json task config")
parser.add_argument("-tn", "--task_name", help="task name: worker/ps/scheduler", default="worker")
parser.add_argument("-ti", "--task_index", help="json task config", default=0, type=int)
parser.add_argument("-tnm", "--task_num", help="json task config", type=int)
parser.add_argument("-ai", "--app_id", help="json task config")
parser.add_argument("-zk", "--zk_addr", help="scheduler zookeeper addr")
parser.add_argument("-rm", "--run_mode", help="running mode:local/distributed")
parser.add_argument("-pn", "--ps_num", help="ps num")
parser.add_argument("-pm", "--ps_memory_m", help="ps memory MB")
parser.add_argument("-pc", "--ps_cpu_cores", help="ps cpu cores")
parser.add_argument("-cp", "--ckpt_dir", help="checkpoint dir")
parser.add_argument("-tt", "--task_type", help="task type", default="train")
parser.add_argument("-ms", "--model_server", help="model server", default="")
parser.add_argument("-msn", "--model_server_num", help="model server num", default="")
parser.add_argument("-f", "--notebook_file", help="notebook file", default="")
_BASE_ARGS, options = parser.parse_known_args()

def get_task_name():
  return _BASE_ARGS.task_name

def get_task_index():
  return _BASE_ARGS.task_index

def get_app_id():
  return _BASE_ARGS.app_id

def get_zk_addr():
  return _BASE_ARGS.zk_addr

def get_run_mode():
  return _BASE_ARGS.run_mode

def get_config():
  return _BASE_ARGS.config

def get_ps_num():
  return _BASE_ARGS.ps_num

def get_ps_memory_m():
  return _BASE_ARGS.ps_memory_m

def get_ps_cpu_cores():
  return _BASE_ARGS.ps_cpu_cores

def get_model_server():
  return _BASE_ARGS.model_server.split(',') if _BASE_ARGS.model_server != "" else []

def get_model_server_num():
  return [int(i) for i in _BASE_ARGS.model_server_num.split(',')] if _BASE_ARGS.model_server_num != "" else []

def get_ckpt_dir():
  if _BASE_ARGS.ckpt_dir is not None:
    return _BASE_ARGS.ckpt_dir
  return get_config("checkpoint", "output_dir")

def get_task_type():
  return _BASE_ARGS.task_type

def get_task_num():
  if _BASE_ARGS.task_num is not None:
    return _BASE_ARGS.task_num
  task_num = get_config("worker", "instance_num")
  if task_num is None:
    return 1
  return task_num

def get_ps_mode():
  if get_config('ps_mode') == False:
    return False
  return True

_BASE_CONFIG = None
def get_config(*keys, **kwargs): 
  default_value = None
  if 'default_value' in kwargs:
    default_value = kwargs['default_value']
  global _BASE_CONFIG 
  if _BASE_CONFIG is None:
    if _BASE_ARGS.config:
      _BASE_CONFIG = json.load(open(_BASE_ARGS.config, 'r'))
      print("config: %s" % str(_BASE_CONFIG))
    else:
      _BASE_CONFIG = {}
  
  # _BASE_CONFIG
  if len(keys) == 0:
    return _BASE_CONFIG
  
  # value
  value = _BASE_CONFIG
  for key in keys:
    if key is None or value is None: 
      return None if not default_value else default_value
    if key in value:
      value = value[key]
    else:
      return None if not default_value else default_value
  return value

