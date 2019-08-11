import xdl
from xdl.python.framework.session import Hook
from xdl.python.training.training_utils import get_global_step
import os
import numpy as np

class GlobalStepMarkHook(Hook):
  def __init__(self, var_name, ids):
    super(GlobalStepMarkHook, self).__init__()
    from xdl.python.sparse_engine.embedding import *
    embedding_info = get_embedding_info_by_name(var_name)
    global_step = get_global_step().value
    with xdl.control_dependencies([embedding_info.embedding]):
      self._update = xdl.ps_mark_op(ids=ids,
                                    i=global_step,
                                    var_name=var_name,
                                    pattern="global_step")

  def before_run(self, v):
      return self._update
  
class GlobalStepFilterHook(Hook):
  def __init__(self, vars, interval_steps, expire_steps):
    super(GlobalStepFilterHook, self).__init__(priority=2999)
    self._interval_steps = interval_steps
    self._expire_steps = expire_steps
    self._vars = vars
    self._global_step = get_global_step()
    self._last_filter_step = 0

    self.gstep_val = 0

  def generate_filter_ops(self, current_step):
    all_ops = []
    for var_name in self._vars:
      filter_op = xdl.hash_filter(var_name, os.path.dirname(os.path.realpath(__file__)) +
        "/filter.py", "filter", {"x":np.array(self._expire_steps), "y":np.array(current_step)})
      all_ops.append(filter_op)
    return all_ops

  def before_run(self, v):
    return [self._global_step.value]

  def after_run(self, v):
    self.gstep_val = v[0] if isinstance(v, list) else v
    if self.gstep_val - self._last_filter_step >= self._interval_steps:
      print("GlobalStepFilterHook running all")
      xdl.execute(self.generate_filter_ops(self.gstep_val))
      self._last_filter_step = self.gstep_val

  def end(self):
    self.gstep_val = xdl.execute(self._global_step.value)
    print("GlobalStepFilterHook running all")
    xdl.execute(self.generate_filter_ops(self.gstep_val))