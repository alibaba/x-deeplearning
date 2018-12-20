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

def fc(inputs, w_shape):
  with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
    w = tf.get_variable("weights", 
                        w_shape, 
                        initializer=tf.constant_initializer(0.1), 
                        regularizer=tf.nn.l2_loss)
  return tf.matmul(inputs, w)

def main():
  dense = xdl.mock_dense_op(shape=[1, 16], value=0.01, name_="dense")
  labels = xdl.mock_dense_op(shape=[1, 1], value=1.0, name_="label")
  ids = xdl.convert_to_tensor(np.array([[0,0], [0,1], [0,2]], dtype=np.int64))
  values = xdl.convert_to_tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
  segments = xdl.convert_to_tensor(np.array([3], dtype=np.int32))
  sparse = xdl.SparseTensor(ids, values, segments)
  emb = xdl.embedding("sparse", sparse, xdl.Ones(), 1, 16, 'sum', vtype='hash')
  loss = model(dense, emb, labels)
  train_op = xdl.SGD(0.5).optimize()
  sess = xdl.TrainSession()
  loss, gradients = sess.run([loss, xdl.get_sparse_grads('sparse').grad])
  return loss, gradients

@xdl.tf_wrapper(is_training=True)
def model(dense, emb, labels):
    fc2 = fc(dense, [16, 1])
    logits = fc2 + emb
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

class GradientTest(unittest.TestCase):
  def test_all(self):
    dense = xdl.mock_dense_op(shape=[1, 16], value=0.01, name_="dense")
    labels = xdl.mock_dense_op(shape=[1, 1], value=1.0, name_="label")
    ids = xdl.convert_to_tensor(np.array([[0,0], [0,1], [0,2]], dtype=np.int64))
    values = xdl.convert_to_tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    segments = xdl.convert_to_tensor(np.array([3], dtype=np.int32))
    sparse = xdl.SparseTensor(ids, values, segments)
    emb = xdl.embedding("sparse", sparse, xdl.Ones(), 1, 16, 'sum', vtype='hash')
    loss = model(dense, emb, labels)
    train_op = xdl.SGD(0.5).optimize()
    sess = xdl.TrainSession()
    _, l, g = sess.run([train_op, loss, xdl.get_sparse_grads('sparse').grad])
    self.assertTrue((l==np.array(0.0024364376, dtype=np.float32)).all())
    self.assertTrue((g==np.array([[-0.002433472],[-0.004866944],[-0.007300416]], dtype=np.float32)).all())
    sparse_var = xdl.get_variable_by_name('sparse')
    weights = sess.run(sparse_var.gather(np.array([[0,0],[0,1],[0,2]], dtype=np.int64)))
    self.assertTrue((weights==np.array([[1.0012168],[1.0024334],[1.0036502]], dtype=np.float32)).all())
    _, l, g = sess.run([train_op, loss, xdl.get_sparse_grads('sparse').grad])
    self.assertTrue((l==np.array(0.002395329, dtype=np.float32)).all())
    self.assertTrue((g==np.array([[-0.0023924622],[-0.0047849244],[-0.0071773864]], dtype=np.float32)).all())
    weights = sess.run(sparse_var.gather(np.array([[0,0],[0,1],[0,2]], dtype=np.int64)))    
    self.assertTrue((weights==np.array([[1.002413],[1.0048258],[1.0072389]], dtype=np.float32)).all())

def suite():
  return unittest.TestLoader().loadTestsFromTestCase(GradientTest)

if __name__ == '__main__':
  unittest.TextTestRunner().run(suite())
