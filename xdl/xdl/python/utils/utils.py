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

import xdl
import numpy as np
import time
from datetime import datetime
from xdl.python.framework.variable import Variable
from xdl.python.lib.datatype import *
from xdl.python.ops.init_ops import *
from xdl.python.utils.config import *
from xdl.python.lib.graph import control_dependencies, execute
from xdl.python.ops.ps_ops import barrier_op_v2

def decode_strings_from_buf(addrs, lens):
  """ get string at index by addrs and lens
    """
  strings = list()
  assert len(addrs) == len(lens)
  for i in range(len(lens)):
    if lens[i] > 0:
      strings.append("".join(map(chr, addrs[i, 0:lens[i]])))
  return strings

def str_to_timestamp(s, format='%Y%m%d%H%M%S', accuracy='us'):
  t = datetime.strptime(s, format)
  ts = int(time.mktime(t.timetuple()))
  if accuracy == 's':
    return ts
  elif accuracy == 'ms':
    return ts * 1000
  elif accuracy == 'us':
    return ts * 1000 * 1000
  else:
    raise RuntimeError('unknown accuracy %s' % accuracy)

def timestamp_to_str(ts, format='%Y%m%d%H%M%S', accuracy='us'):
  if accuracy == 'us':
    ts = ts / 1000 / 1000
  elif accuracy == 'ms':
    ts = ts / 1000
  t = datetime.fromtimestamp(ts)
  return datetime.strftime(t, format)

BARRIER_VARIABLES = {}
__BARRIER_TOKEN__ = int(time.time())

def add_barrier_variable(name):
  global BARRIER_VARIABLES
  if name not in BARRIER_VARIABLES:
    BARRIER_VARIABLES[name] = Variable(
        name,
        dtype=DataType.int64,
        shape=[get_task_num()],
        initializer=Zeros(),
        trainable=False
    )

def barrier_with_timestamp(name, timestamp, token=None):
  var = BARRIER_VARIABLES[name]
  update_op = xdl.ps_sparse_assign_op(
      var_name=var.name,
      var_type=var.vtype,
      ids=np.array([get_task_index()], dtype=np.int32),
      values=np.array([timestamp], dtype=np.int64)
  )
  with control_dependencies([update_op]):
    barrier_op = barrier_op_v2(get_task_index(), get_task_num(), token)
  with control_dependencies([barrier_op]):
    pull_op = xdl.ps_pull_op(
        var_name=var.name, var_type=var.vtype, dtype=var.dtype
    )
  return np.max(execute(pull_op))
