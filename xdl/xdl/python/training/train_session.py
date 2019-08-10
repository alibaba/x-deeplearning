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

import time
import xdl
import numpy as np
import math
from xdl.python.lib.graph import execute
from xdl.python.training.env import current_env, is_local_mode
from xdl.python.framework.session import Hook, Session
from xdl.python.framework.variable import global_variables, variable_registers, global_initializers
from xdl.python.lib.error import PsError, InternalError, OutOfRange
from xdl.python.utils.ps_utils import restart_client
from xdl.python.utils.metrics import add_metrics, get_all_metrics
from xdl.python.training.training_utils import get_global_step
from xdl.python.training.saver import Saver
from xdl.python.lib.graph import current_graph
from xdl.python.backend.model_scope import cur_model_scope, model_scope, get_model_scopes
from xdl.python.utils.config import get_ckpt_dir
from xdl.python.utils.collections import get_collection, READER_HOOKS, reset_collections

def _restart_client(retry_cnt=3, interval=10):
    i = 0
    while (i < retry_cnt and not restart_client()):
        i += 1
        time.sleep(interval)
        if i >= 3:
            raise InternalError("restart client failed")

class SyncRunHook(Hook):
    def __init__(self, index, worker_count):
        super(SyncRunHook, self).__init__()
        self._index = index
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

    def end(self):
        execute(xdl.worker_report_finish_op(id = np.array(self._index, dtype=np.int32)))

class WorkerFinishHook(Hook):
    def __init__(self, index, worker_count, finish_rate):
        super(WorkerFinishHook, self).__init__()
        self._is_chief = False
        if index == 0:
            self._is_chief = True
        self._index = index
        self._worker_count = worker_count
        self._op = xdl.get_worker_finish_count_op()
        self._finish_rate = finish_rate

    def before_run(self, v):
        if self._is_chief:
            return [self._op]
        else:
            return []

    def after_run(self, v):
        if self._is_chief and v[0] >= math.ceil(self._finish_rate * self._worker_count / 100.0):
            print("Finish_num is [%ld] limit is [%d], request_stop" % (v[0], math.ceil(self._finish_rate * self._worker_count / 100.0)))
            raise OutOfRange("WorkerFinished")

    def end(self):
        execute(xdl.worker_report_finish_op(id = np.array(self._index, dtype=np.int32)))
        if self._is_chief:
            finish_num = execute(self._op)
            while finish_num < math.ceil(self._finish_rate * self._worker_count / 100.0):
                print("Finish_num is [%ld] limit is [%d], waiting..." % (finish_num, math.ceil(self._finish_rate * self._worker_count / 100.0)))
                finish_num = execute(self._op)
                time.sleep(10)
            print("Finish_num is [%ld] limit is [%d], exiting..." % (finish_num, math.ceil(self._finish_rate * self._worker_count / 100.0)))

class SemiSyncRunHook(Hook):
    def __init__(self, index, worker_count, staleness=0):
        super(SemiSyncRunHook, self).__init__()
        self._index = index
        self._before = xdl.ps_asynchronize_enter_op(
            np.array(index, dtype=np.int32),
            np.array(staleness, dtype=np.int32),
            np.array(worker_count, dtype=np.int32))

    def before_run(self, v):
        execute([self._before])
        return []

    def end(self):
        execute(xdl.worker_report_finish_op(id = np.array(self._index, dtype=np.int32)))

class BarrierHook(Hook):
    def __init__(self, index, worker_count):
        super(BarrierHook, self).__init__()
        self._before = xdl.worker_barrier_op(
            np.array(index, dtype=np.int32),
            np.array(worker_count, dtype=np.int32))

    def before_run(self, v):
        execute([self._before])
        return []

class MetricsHook(Hook):
    def __init__(self, name, op, interval):
        super(MetricsHook, self).__init__(priority=1003)
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
            add_metrics(self._name, '')
            return
        add_metrics(self._name, str(v))

class MetricsPrinterHook(Hook):
    def __init__(self, log_fmt, interval=100, file_path=''):
        super(MetricsPrinterHook, self).__init__()
        self._log_fmt = log_fmt
        self._interval = interval
        self._counter = 0
        self._protocol, self._hostname, self._path = self.parse_path(file_path)
        if self._protocol == 'local':
            self._local_file = open(self._path, 'w')
        elif self._protocol == 'hdfs':
            import pyhdfs
            self._hdfs_client = pyhdfs.HdfsClient(hosts=self._hostname.replace(':', ','))
            self._hdfs_file = self._hdfs_client.create(self._path, '', overwrite=True)
        elif self._protocol != '':
            raise ValueError('not supportted file system:%s' % self._protocol)

    def parse_path(self, file_path):
        if file_path == '':
            return '', '', ''
        protocol = ''
        hostname = ''
        path = ''
        pos = file_path.find('://')
        if pos != -1:
            protocol = file_path[:pos]
            pos2 = file_path.find('/', pos + 3)
            if pos2 != -1:
                hostname = file_path[pos+3:pos2]
                path = file_path[pos2:]
            else:
                hostname = file_path[pos+3:]
        else:
            protocol = 'local'
            path = file_path
        return protocol, hostname, path

    def before_run(self, v):
        self._counter = self._counter + 1
        return []

    def after_run(self, v):
        if self._counter % self._interval == 0:
            msg = self._log_fmt % get_all_metrics()
            print(msg)
            if self._path != '':
                if self._protocol == 'local':
                    self._local_file.write(msg + '\n')
                elif self._protocol == 'hdfs':
                    self._hdfs_client.append(self._path, msg + '\n')
                else:
                    pass

class QpsMetricsHook(Hook):
    def __init__(self):
        super(QpsMetricsHook, self).__init__()
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
            self._times.append(int(round(time.time() * 1000)))
        else:
            self._times.pop(0)
            self._times.append(int(round(time.time() * 1000)))

        add_metrics("lstep", self._lstep)
        add_metrics("gstep", self._gstep)
        if len(self._times) > 1:
            interval = self._times[-1] - self._times[0]
            if interval == 0:
                interval = 1
            gsteps = self._gsteps[-1] - self._gsteps[0]
            lsteps = self._lsteps[-1] - self._lsteps[0]
            gqps = gsteps * 1000.0 / interval
            lqps = lsteps * 1000.0 / interval
            add_metrics("gqps", '%.1f' % gqps)
            add_metrics("lqps", '%.1f' % lqps)
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
            print('run ops:', ops, ' fail, retry cnt:', i)
            time.sleep(10)
            _restart_client()


class WorkerHook(Hook):
    def __init__(self):
        super(WorkerHook, self).__init__(priority=1002)

    def create_session(self):
        scopes = list(get_model_scopes())
        if global_variables(scopes) is None or \
                len(global_variables(scopes)) == 0:
            return
        execute_with_retry(variable_registers(scopes))
        var_set = set(global_variables(scopes))
        while len(var_set) > 0:
            for var in global_variables(scopes):
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
    def __init__(self):
        super(ChiefHook, self).__init__(priority=1001)

    def create_session(self):
        scopes = list(get_model_scopes())
        if global_variables(scopes) is None or\
                len(global_variables(scopes)) == 0:
            return
        execute_with_retry(variable_registers(scopes))
        execute_with_retry(global_initializers(scopes))


class LoggerHook(Hook):
    def __init__(self, ops, fmt, interval=1):
        super(LoggerHook, self).__init__()
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
            #self._hooks = self._hooks + [ChiefHook(), WorkerHook()]
            self._hooks = self._hooks + [ChiefHook()]
        else:
            self._hooks = self._hooks + [WorkerHook()]
        def take_priority(elem):
            return elem._priority
        self._hooks.sort(key=take_priority)
        self._session = Session(self._hooks)

    def run(self, v, run_option=None, run_statistic=None):
        return self._session.run(v, run_option, run_statistic)

    def end(self):
        return self._session.end()


class TrainSession(object):
    def __init__(self, hooks=None):
        current_env().sess_start()
        self._hooks = [] if hooks is None else hooks
        reader_hooks = get_collection(READER_HOOKS)
        if reader_hooks is not None:
            self._hooks.extend(reader_hooks)
        self._cur_scope = cur_model_scope()
        self._session = SimpleSession(hooks)
        self._finish = False

    def run(self, v, run_option=None, run_statistic=None):
        current_graph().unfinalize()
        try:
            if self._session is None:
                _restart_client()
                with model_scope(self._cur_scope):
                    self._session = SimpleSession(self._hooks)
            return self._session.run(v, run_option, run_statistic)
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

    def end(self):
        self._session.end()
        self._finish = True
        
    def should_stop(self):
        return self._finish
