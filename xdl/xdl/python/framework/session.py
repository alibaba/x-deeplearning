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

from xdl.python.lib.graph import execute

class Hook(object):
  def __init__(self, priority = 2000):
    self._priority = priority

  def create_session(self):
    pass

  def run(self, v):
    return self.before_run(v), self.after_run
  
  def before_run(self, v):
    return []

  def after_run(self, v):
    return None

  def end(self):
    pass

class Session(object):
  def __init__(self, hooks = None):
    if hooks is None:
      hooks = []
    self._hooks = hooks
    self._create_session()

  def _create_session(self):
    for hook in self._hooks:
      hook.create_session()

  def run(self, v, run_option=None, run_statistic=None):
    run_item = [v]
    cbs = []
    for hook in self._hooks:
      run, cb = hook.run(v)
      run_item.append(run)
      cbs.append(cb)
    results = execute(run_item, run_option, run_statistic)
    for i in range(len(cbs)):
      cbs[i](results[i + 1])
    return results[0]

  def end(self):
    for hook in self._hooks:
      hook.end()
    
