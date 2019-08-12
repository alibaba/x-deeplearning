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

import sys
from xdl.python.lib.error import ArgumentError
from xdl.python.training.train_session import TrainSession, MetricsHook, QpsMetricsHook, MetricsPrinterHook
from xdl.python.training.saver import CheckpointHook
from xdl.python.ops.auc import auc, batch_auc, reset_auc_variables_op
from xdl.python.utils.collections import get_collection, READER_HOOKS, reset_collections
from xdl.python.backend.model_scope import model_scope
from xdl.python.utils.config import get_task_index

LOG_FMT = "lstep[%(lstep)s] gstep[%(gstep)s] lqps[%(lqps)s] gqps[%(gqps)s] auc[%(auc)s]"
EVAL_LOG_FMT = "lstep[%(lstep)s] lqps[%(lqps)s] auc[%(auc)s]"
PREDICT_LOG_FMT = "lstep[%(lstep)s] lqps[%(lqps)s] prediction[%(prediction)s]"

class Estimator(object):
  """Estimator is used to simplify train, evaluate and predict a model
  Args:
  model_fn: a funtion which defines a model, must return loss/logits as
  as first/second return value
  optimizer: a xdl optimizer decide how to update gradient
  checkpoint_dir: model save path
  """
  def __init__(self, 
               model_fn=None, 
               optimizer=None):
    self._model_fn = model_fn
    self._optimizer = optimizer
    
  def train(self, input_fn, 
            auc_fn=auc,
            auc_interval=100,
            auc_bucket_num=200,
            max_step=sys.maxint, 
            checkpoint_interval=None,
            log_format=LOG_FMT,
            user_hooks=None):
      '''
      Args:
      input_fn:
      auc_fn:
      auc_interval:
      max_step:
      checkpoint_interval:
      log_format:
      user_hooks:
      '''
      data, labels = input_fn()
      model_outputs = self._model_fn(data, labels)
      if len(model_outputs) < 2:
        raise ArgumentError("model_fn must return loss and logits")
      loss = model_outputs[0]
      logits = model_outputs[1]
      train_op = self._optimizer.optimize()
      auc_op = auc_fn(logits, labels, num_thresholds=auc_bucket_num)
      hooks = []
      hooks.append(QpsMetricsHook())
      hooks.append(MetricsHook("auc", auc_op, interval=auc_interval))
      if user_hooks is not None:
        if isinstance(user_hooks, list):
          hooks.extend(user_hooks)
        else:
          hooks.append(user_hooks)
      reader_hooks = get_collection(READER_HOOKS)
      if reader_hooks is not None:
        hooks.extend(reader_hooks)
      if checkpoint_interval and get_task_index() == 0:
        hooks.append(CheckpointHook(checkpoint_interval))
      hooks.append(MetricsPrinterHook(log_format, auc_interval))
      sess = TrainSession(hooks=hooks)
      i = 0
      while not sess.should_stop() and i < max_step:
        sess.run(train_op)
        i = i + 1

  def evaluate(self, input_fn, 
               checkpoint_version="", 
               log_format=EVAL_LOG_FMT, 
               log_interval=100,
               max_step=sys.maxint,
               auc_fn=auc,
               auc_bucket_num=200,
               user_hooks=None):
    '''
    Args:
    input_fn:
    checkpoint_version:
    log_format:
    log_interval:
    max_step:
    auc_fn:
    user_hooks:
    '''
    from xdl.python.training.saver import Saver
    if get_task_index() == 0:
      saver = Saver()
      saver.restore(checkpoint_version)
    data, labels = input_fn()
    model_outputs = self._model_fn(data, labels)
    if len(model_outputs) < 2:
      raise ArgumentError("model_fn must return loss and logits")
    logits = model_outputs[1]
    auc_op = auc_fn(logits, labels, num_thresholds=auc_bucket_num)
    hooks = []
    hooks.append(QpsMetricsHook())
    hooks.append(MetricsHook("auc", auc_op, interval=1))
    if user_hooks is not None:
      if isinstance(user_hooks, list):
        hooks.extend(user_hooks)
      else:
        hooks.append(user_hooks)
    hooks.append(MetricsPrinterHook(log_format, log_interval))        
    sess = TrainSession(hooks=hooks)
    if id(auc_fn) == id(auc):
      sess.run(reset_auc_variables_op(auc_bucket_num))
    i = 0
    while not sess.should_stop() and i < max_step:
      sess.run([])
      i = i + 1

  def predict(self, input_fn, 
              checkpoint_version="", 
              log_format=PREDICT_LOG_FMT, 
              log_interval=100,
              max_step=sys.maxint,
              user_hooks=None):
    ''' 
    Args:
    input_fn:
    checkpoint_version:
    log_format:
    log_interval:
    max_step:
    user_hooks:
    '''
    from xdl.python.training.saver import Saver
    if get_task_index() == 0:
      saver = Saver()
      saver.restore(checkpoint_version)
    data, labels = input_fn()
    model_outputs = self._model_fn(data, labels)
    if len(model_outputs) < 2:
      raise ArgumentError("model_fn must return loss and logits")
    logits = model_outputs[1]
    hooks = []
    hooks.append(QpsMetricsHook())
    hooks.append(MetricsHook("prediction", logits, interval=1))
    if user_hooks is not None:
      if isinstance(user_hooks, list):
        hooks.extend(user_hooks)
      else:
        hooks.append(user_hooks)
    hooks.append(MetricsPrinterHook(log_format, log_interval))
    sess = TrainSession(hooks=hooks)
    i = 0
    while not sess.should_stop() and i < max_step:
      sess.run([])
      i = i + 1

  def train_and_evaluate(self, 
                         train_input_fn, 
                         eval_input_fn,
                         eval_interval,
                         eval_steps,
                         checkpoint_interval,
                         auc_fn=auc,
                         auc_bucket_num=200,
                         train_hooks=None,
                         eval_hooks = None,
                         auc_interval=100,
                         log_interval=100,
                         log_format=LOG_FMT,
                         eval_log_format=EVAL_LOG_FMT,
                         max_step=sys.maxint):
    with model_scope('train'):
      datas, labels = train_input_fn()
      train_outputs = self._model_fn(datas, labels)
      if len(train_outputs) < 2:
        raise ArgumentError("model_fn must return loss and logits")
      loss = train_outputs[0]
      logits = train_outputs[1]
      train_op = self._optimizer.optimize()
      auc_op = auc_fn(logits, labels, num_thresholds=auc_bucket_num, namescope="train_auc")
  
      train_hooks = []
      train_hooks.append(QpsMetricsHook())
      train_hooks.append(MetricsHook("auc", auc_op, interval=auc_interval))
      if train_hooks is not None:
        if isinstance(train_hooks, list):
          train_hooks.extend(train_hooks)
        else:
          train_hooks.append(train_hooks)
      reader_hooks = get_collection(READER_HOOKS)
      if reader_hooks is not None:
        train_hooks.extend(reader_hooks)
      if checkpoint_interval and get_task_index() == 0:
        train_hooks.append(CheckpointHook(checkpoint_interval))
      train_hooks.append(MetricsPrinterHook(log_format, auc_interval))
      train_sess = TrainSession(hooks=train_hooks)

    with model_scope('test'):
      eval_datas, eval_labels = eval_input_fn()
      eval_outputs = self._model_fn(eval_datas, eval_labels)
      if len(eval_outputs) < 2:
        raise ArgumentError("model_fn must return loss and logits")
      eval_logits = eval_outputs[1]
      eval_auc_op = auc_fn(eval_logits, eval_labels, num_thresholds=auc_bucket_num, namescope="eval_auc")
      eval_hooks = []
      eval_hooks.append(QpsMetricsHook())
      eval_hooks.append(MetricsHook("auc", eval_auc_op, interval=1))
      if eval_hooks is not None:
        if isinstance(eval_hooks, list):
          eval_hooks.extend(eval_hooks)
        else:
          eval_hooks.append(eval_hooks)
      eval_hooks.append(MetricsPrinterHook(eval_log_format, log_interval))        
      eval_sess = TrainSession(hooks=eval_hooks)

    lstep = 0
    while True:
      print('\n>>> start train at local step[%d]\n' % lstep)
      while not train_sess.should_stop() and (lstep == 0 or lstep % eval_interval != 0) \
            and lstep < max_step:
        train_sess.run(train_op)
        lstep = lstep + 1
      lstep = lstep + 1
      eval_step = 0
      print('\n>>> start evaluate at local step[%d]\n' % lstep)
      while not eval_sess.should_stop() and eval_step < eval_steps:
        eval_sess.run([])
        eval_step = eval_step + 1
      if train_sess.should_stop() or lstep >= max_step:
        break
            
        
        
        
        


