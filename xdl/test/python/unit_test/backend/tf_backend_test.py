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

import xdl
import unittest
import numpy as np
from xdl.python.lib.datatype import *
from xdl.python.lib.graph import execute
try:
  from xdl.python.backend.tf.tf_backend import *
except ImportError:
  sys.exit(0)


def fc(inputs, w_shape, b_shape):
  with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
    w = tf.get_variable("weights", 
                        w_shape, 
                        initializer=tf.constant_initializer(0.1), 
                        regularizer=tf.nn.l2_loss)
  return tf.matmul(inputs, w)

def main():
  dense = xdl.mock_dense_op(shape=[1, 16], value=0.01, name_="dense")
  gear = xdl.mock_dense_op(shape=[1, 1], value=0.01, name_="gear")
  labels = xdl.mock_dense_op(shape=[1, 1], value=1.0, name_="label")
  ids, values, segments = xdl.mock_sparse_op(dense_shape=[1, 16], name_="wide")
  sparse = xdl.SparseTensor(ids, values, segments)
  emb = xdl.embedding("sparse", sparse, xdl.Ones(), 1, 16, 'sum')
  gear.set_shape([None, 1])
  dense.set_shape([None, 16])
  labels.set_shape([None, 1])
  with xdl.model_scope("ams_main"):
    loss = ams_main(main_model)(dense, emb, labels, gear_inputs=[gear])
    sess = xdl.TrainSession()
    return sess.run(xdl.get_collection("gear_grad"))

def gear():
  forward = xdl.mock_dense_op(shape=[1, 16], value=0.01, name_="forward")
  backward = xdl.mock_dense_op(shape=[1, 16], value=0.02, name_="backward")
  labels = xdl.mock_dense_op(shape=[1, 1], value=1.0, name_="label1")
  init_grad = xdl.mock_dense_op(shape=[1, 1], value=0.3, name_="init_grad")
  forward.set_shape([None, 16])
  backward.set_shape([None, 16])
  labels.set_shape([None, 1])  
  init_grad.set_shape([None, 1])  
  predict = ams_gear([forward], [backward], init_grad)(gear_model)(None)
  with xdl.model_scope("ams_gear_forward"):
    sess = xdl.TrainSession()
    prediction = sess.run(predict)
  with xdl.model_scope("ams_gear_backward"):
    grads = xdl.get_gradient("weights")
    sess = xdl.TrainSession()
    weight_grads = sess.run(grads)
    return prediction, weight_grads

def main_model(dense, emb, labels, gear_inputs):
  fc2 = fc(dense, [16, 1], [1])
  logits = fc2 + emb + gear_inputs[0]
  return logits

def gear_model(inputs):
  logits = fc(inputs, [16,1], [1])
  return logits

class TFBackendTest(unittest.TestCase):
  def test_ams_main(self):
    gear_gradient = main()
    self.assertTrue(gear_gradient[0]==np.array([[1.0]], dtype=np.float32))

  def test_ams_gear(self):
    prediction, grad = gear()
    self.assertTrue((prediction==np.array([[0.016]], dtype=np.float32)).all())
    self.assertTrue((grad==np.repeat(np.array([0.006], dtype=np.float32), 16).reshape(1,16)).all())

def suite():
  return unittest.TestLoader().loadTestsFromTestCase(TFBackendTest)

if __name__ == '__main__':
  unittest.TextTestRunner().run(suite())
