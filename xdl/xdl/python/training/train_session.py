# Copyright (C) 2016-2018 Alibaba Group Holding Limited
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

import time
import datetime
import xdl
import numpy as np
from xdl.python.lib.graph import execute
from xdl.python.training.env import current_env, is_local_mode
from xdl.python.framework.session import Hook, Session
from xdl.python.framework.variable import global_variables, variable_registers, global_initializers
from xdl.python.lib.error import PsError, InternalError, OutOfRange
from xdl.python.utils.ps_utils import restart_client
from xdl.python.utils.metrics import add_metrics, get_all_metrics
from xdl.python.training.training_utils import get_global_step
from xdl.python.lib.graph import current_graph
from xdl.python.backend.model_scope import cur_model_scope


class SyncRunHook(Hook):
    def __init__(self, index, worker_count):
        self._before = xdl.ps_synchronize_enter_op(
            np.array(index, dtype=np.int32), np.array(worker_count, dtype=np.int32))
        self._after = xdl.ps_synchronize_leave_op(
            np.array(index, dtype=np.int32))

    def before_run(self, v):
        execute([self._before])
        return []

    def after_run(self, v):
        execute([self._after])
        return None


class SemiSyncRunHook(Hook):
    def __init__(self, index, worker_count, staleness=0):
        self._before = xdl.ps_asynchronize_enter_op(
            np.array(index, dtype=np.int32),
            np.array(staleness, dtype=np.int32),
            np.array(worker_count, dtype=np.int32))

    def before_run(self, v):
        execute([self._before])
        return []

class BarrierHook(Hook):
    def __init__(self, index, worker_count):
        self._before = xdl.worker_barrier_op(
            np.array(index, dtype=np.int32),
            np.array(worker_count, dtype=np.int32))

    def before_run(self, v):
        execute([self._before])
        return []

class MetricsHook(Hook):
    def __init__(self, name, op, interval):
        self._op = op
        self._name = name
        self._interval = interval
        self._counter = 0

    def before_run(self, v):
        self._counter = self._counter + 1
        if self._counter % self._interval == 0:
            return self._op
        else:
            return []

    def after_run(self, v):
        if isinstance(v, list) and len(v) == 0:
            return
        add_metrics(self._name, str(v))


class MetricsPrinterHook(Hook):
    def __init__(self, log_fmt, interval=100):
        self._log_fmt = log_fmt
        self._interval = interval
        self._counter = 0

    def before_run(self, v):
        return []

    def after_run(self, v):
        self._counter = self._counter + 1
        if self._counter % self._interval == 0:
            print(self._log_fmt % get_all_metrics())


class QpsMetricsHook(Hook):
    def __init__(self):
        self._lstep = 0
        self._gstep = 0
        self._lsteps = []
        self._gsteps = []
        self._times = []
        self._interval = 20

    def before_run(self, v):
        return get_global_step().value

    def after_run(self, v):
        self._lstep = self._lstep + 1
        self._gstep = v
        if len(self._lsteps) < self._interval:
            self._lsteps.append(self._lstep)
        else:
            self._lsteps.pop(0)
            self._lsteps.append(self._lstep)
        if len(self._gsteps) < self._interval:
            self._gsteps.append(self._gstep)
        else:
            self._gsteps.pop(0)
            self._gsteps.append(self._gstep)
        if len(self._times) < self._interval:
            dt = datetime.datetime.now()
            self._times.append(dt.microsecond + dt.second * 1000000)
        else:
            self._times.pop(0)
            dt = datetime.datetime.now()
            self._times.append(dt.microsecond + dt.second * 1000000)

        add_metrics("lstep", self._lstep)
        add_metrics("gstep", self._gstep)
        if len(self._times) > 1:
            interval = self._times[-1] - self._times[0]
            if interval == 0:
                interval = 1
            gsteps = self._gsteps[-1] - self._gsteps[0]
            lsteps = self._lsteps[-1] - self._lsteps[0]
            gqps = int(gsteps * 1000000.0 / interval)
            lqps = int(lsteps * 1000000.0 / interval)
            add_metrics("gqps", str(gqps))
            add_metrics("lqps", str(lqps))
        else:
            add_metrics("gqps", "0")
            add_metrics("lqps", "0")


def execute_with_retry(ops, retry_cnt=6):
    ops = list(ops) if isinstance(ops, (list, tuple)) else [ops]
    i = 0
    while i < retry_cnt:
        try:
            return execute(ops)
        except (PsError) as e:
            i = i + 1
            if i == retry_cnt:
                raise e
            print('execute fail retry cnt[%d]' % i)
            time.sleep(30)
            restart_client()


class WorkerHook(Hook):
    def create_session(self):
        if global_variables() is None or len(global_variables()) == 0:
            return
        execute_with_retry(variable_registers())
        var_set = set(global_variables())
        while len(var_set) > 0:
            for var in global_variables():
                if var not in var_set:
                    continue
                inited = execute_with_retry([var.is_initialized_op])
                if inited == [1]:
                    var_set.remove(var)
                else:
                    print('waiting for initialize variable[%s]' % var.name)
            if is_local_mode():
                time.sleep(0.1)
            else:
                time.sleep(3)


class ChiefHook(Hook):
    def create_session(self):
        if global_variables() is None or\
                len(global_variables()) == 0:
            return
        execute_with_retry(variable_registers())
        execute_with_retry(global_initializers())


class LoggerHook(Hook):
    def __init__(self, ops, fmt, interval=1):
        if not isinstance(ops, list):
            self._ops = [ops]
        else:
            self._ops = ops
        self._fmt = fmt
        self._interval = interval
        self._step = 0

    def before_run(self, v):
        return self._ops

    def after_run(self, v):
        self._step = self._step + 1
        if self._step % self._interval != 0:
            return
        if len(self._ops) != len(v):
            raise Exception("sess run outputs size error!")
        msg = self._fmt
        for i in range(len(v)):
            msg = msg.replace("{%d}" % i, str(v[i]))
        print(msg)


class SimpleSession(object):
    def __init__(self, hooks=None):
        env = current_env()
        if env is None:
            self._is_chief = True
        else:
            self._is_chief = env.is_chief
        if hooks is None:
            hooks = []
        self._hooks = hooks
        if self._is_chief:
            self._hooks = [ChiefHook(), WorkerHook()] + self._hooks
        else:
            self._hooks = [WorkerHook()] + self._hooks
        self._session = Session(self._hooks)

    def run(self, v, run_option=None, run_statistic=None, feed_dict=None):
        return self._session.run(v, run_option, run_statistic, feed_dict=feed_dict)

    def end(self):
        return self._session.end()

class TrainSession(object):
    def __init__(self, hooks=None):
        current_env().sess_start()
        self._hooks = hooks
        self._session = SimpleSession(hooks)
        self._finish = False
#    current_graph().finalize()

    def _restart_client(self, retry_cnt=3, interval=10):
        i = 0
        while (i < retry_cnt and not restart_client()):
            i += 1
            time.sleep(interval)
            if i >= 3:
                raise InternalError("restart client failed")

    def run(self, v, run_option=None, run_statistic=None, feed_dict=None):
        current_graph().unfinalize()
        try:
            if self._session is None:
                self._restart_client()
                self._session = SimpleSession(self._hooks)
            return self._session.run(v, run_option, run_statistic, feed_dict=feed_dict)
        except (PsError) as e:
            print('An error was raised. This may be due to a preemption in '
                  'a connected worker or parameter server. The current '
                  'session will be closed and a new session will be '
                  'created. Error: %s', str(e))
            time.sleep(5)
            self._session = None
        except OutOfRange:
            self._session.end()
            self._finish = True
            return None

    def should_stop(self):
        return self._finish
