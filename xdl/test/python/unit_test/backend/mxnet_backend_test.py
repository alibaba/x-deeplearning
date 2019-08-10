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
import unittest
import numpy as np
from xdl.python.lib.datatype import *
from xdl.python.lib.graph import execute
from xdl.python.backend.mxnet.mxnet_backend import *

def main():
  dense = xdl.mock_dense_op(shape=[1, 16], value=0.01, name_="dense")
  gear = xdl.mock_dense_op(shape=[1, 1], value=0.01, name_="gear")
  labels = xdl.mock_dense_op(shape=[1, 1], value=1.0, name_="label")
  gear.set_shape([1, 1])
  dense.set_shape([1, 16])
  labels.set_shape([1, 1])
  with xdl.model_scope("ams_main"):
    loss = ams_main(main_model)(dense, labels, gear_inputs=[gear])
    sess = xdl.TrainSession()
    return sess.run([xdl.get_collection("gear_grad")])

def gear():
  forward = xdl.mock_dense_op(shape=[1, 16], value=0.01, name_="forward")
  backward = xdl.mock_dense_op(shape=[1, 16], value=0.02, name_="backward")
  labels = xdl.mock_dense_op(shape=[1, 1], value=1.0, name_="label1")
  init_grad = xdl.mock_dense_op(shape=[1, 1], value=0.3, name_="init_grad")
  forward.set_shape([1, 16])
  backward.set_shape([1, 16])
  labels.set_shape([1, 1])  
  init_grad.set_shape([1, 1])  
  predict = ams_gear([forward], [backward], init_grad)(gear_model)(None)
  with xdl.model_scope("ams_gear_forward"):
    sess = xdl.TrainSession()
    prediction = sess.run(predict)
  with xdl.model_scope("ams_gear_backward"):
    grads = xdl.get_gradient("fc_weight")
    sess = xdl.TrainSession()
    fc_weight_grad = sess.run(grads)
    return prediction, fc_weight_grad

def main_model(dense, label, gear_inputs):
  weight = mx.sym.var(name='fc_weight', init=mx.init.Constant(0.1))
  fc = mx.symbol.FullyConnected(data=dense, num_hidden=1, weight=weight, name="fc")
  logits = fc + gear_inputs[0]
  return mx.sym.MakeLoss(logits)

def gear_model(inputs):
  weight = mx.sym.var(name='fc_weight', init=mx.init.Constant(0.1))
  fc = mx.sym.FullyConnected(data=inputs, num_hidden=1, name='fc', weight=weight)
  return fc

class MxnetBackendTest(unittest.TestCase):
  def test_ams_main(self):
    gear_gradient = main()
    self.assertTrue(gear_gradient[0]==np.array([[1.0]], dtype=np.float32))

  def test_ams_gear(self):
    prediction, grad = gear()
    self.assertTrue((prediction==np.array([[0.016]], dtype=np.float32)).all())
    self.assertTrue((grad==np.repeat(np.array([0.006], dtype=np.float32), 16).reshape(1,16)).all())

def suite():
  return unittest.TestLoader().loadTestsFromTestCase(MxnetBackendTest)

if __name__ == '__main__':
  unittest.TextTestRunner().run(suite())
