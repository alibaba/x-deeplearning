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
import time
import datetime
import os
import ctypes
import numpy as np
import tensorflow as tf
import unittest
from xdl.python.utils.collections import READER_HOOKS, get_collection
from xdl.python.utils.metrics import add_metrics
from xdl.python.training.train_session import QpsMetricsHook, MetricsPrinterHook
from xdl.python.framework.variable import trainable_variables_with_scope
from xdl.python.sparse_engine.embedding import is_embedding_var
from xdl.python.backend.model_scope import cur_model_scope
from xdl.python.training.gradient_utils import get_gradient, get_var_mapping

embed_dim = 3
lr = 0.5
is_training = True

@xdl.tf_wrapper(is_training=True)
def model(embs, labels):
    din = tf.concat(embs, axis=-1)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=din, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def mock_embedding(name1, name2):
    ids = xdl.convert_to_tensor(np.array([[0,0], [0,1], [0,2]], dtype=np.int64))
    values = xdl.convert_to_tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    segments = xdl.convert_to_tensor(np.array([3], dtype=np.int32))
    sparse = xdl.SparseTensor(ids, values, segments)
    emb = xdl.embedding(name1, sparse, xdl.Ones(), embed_dim, 16, 'sum', vtype='hash')
    emb.set_shape((1,3))

    ids2 = xdl.convert_to_tensor(np.array([[0,1], [0,2], [1,1]], dtype=np.int64))
    values2 = xdl.convert_to_tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    segments2 = xdl.convert_to_tensor(np.array([3], dtype=np.int32))
    sparse2 = xdl.SparseTensor(ids2, values2, segments2)
    emb2 = xdl.embedding(name2, sparse2, xdl.Ones(), embed_dim, 16, 'sum', vtype='hash')
    emb2.set_shape((1,3))
    return [emb, emb2]

def run(name1, name2, scope, optimizer):
    with xdl.model_scope(scope):
        labels = xdl.mock_dense_op(shape=[1, 1], value=1.0)
        mock_embs = mock_embedding(name1, name2)
        loss = model(mock_embs,labels)
        if optimizer == 'sgd':
            train_op = xdl.SGD(0.5).optimize()
        elif optimizer == 'momentum':
            train_op = xdl.Momentum(0.005, 0.99).optimize()
        elif optimizer == 'ftrl':
            train_op = xdl.Ftrl(0.01).optimize()
        elif optimizer == 'adam':
            train_op = xdl.Adam(0.001).optimize()
        elif optimizer == 'adagrad':
            train_op = xdl.Adagrad(0.04, 0.1).optimize()
        elif optimizer == 'rmsprop':
            train_op = xdl.RMSProp(0.001).optimize()
        else:
            train_op = xdl.SGD(0.5).optimize()
        hooks = []
        sess = xdl.TrainSession(hooks)
        run_ops = [train_op, loss]
        op_names = ['none', 'loss']

        embed_vars = [var for var in trainable_variables_with_scope(scope) if is_embedding_var(var)]
        sparse_embed_grads = []
        for var in embed_vars:
            sparse_embed_grads.append(xdl.get_sparse_grads(var.name))
            op_names.append(var.name + '.indices')
            op_names.append(var.name + '.grads')
        for i in range(len(sparse_embed_grads)):
            run_ops.append(sparse_embed_grads[i].indices)
            run_ops.append(sparse_embed_grads[i].grad)
        var_list = sess.run(run_ops)
        if name1 != name2:
            return var_list[3], var_list[5]
        return var_list[3]

class GradientAggregationTest(unittest.TestCase):
    def test(self):
        id1 = [[0,0],[0,1],[0,2]]
        id2 = [[0,1],[1,1],[0,2]]
        grad1, grad2 = run("853", "861", "normal", "sgd")
        grad3 = run("872", "872", "aggregate", "sgd")
        print 'grad1: ', grad1
        print 'grad2: ', grad2
        print 'grad3: ', grad3
        self.assertTrue(np.allclose(grad1[1], grad3[1]))
        self.assertTrue(np.allclose(grad1[2] + grad2[1], grad3[2]))
        self.assertTrue(np.allclose(grad1[0] + grad2[2], grad3[0]))
        self.assertTrue(np.allclose(grad2[0], grad3[3]))
        grad1, grad2 = run("1", "2", "normal1", "momentum")
        grad3 = run("3", "3", "aggregate1", "momentum")
        self.assertTrue(np.allclose(grad1[1], grad3[1]))
        self.assertTrue(np.allclose(grad1[2] + grad2[1], grad3[2]))
        self.assertTrue(np.allclose(grad1[0] + grad2[2], grad3[0]))
        self.assertTrue(np.allclose(grad2[0], grad3[3]))
        grad1, grad2 = run("4", "5", "normal2", "ftrl")
        grad3 = run("6", "6", "aggregate2", "ftrl")
        self.assertTrue(np.allclose(grad1[1], grad3[1]))
        self.assertTrue(np.allclose(grad1[2] + grad2[1], grad3[2]))
        self.assertTrue(np.allclose(grad1[0] + grad2[2], grad3[0]))
        self.assertTrue(np.allclose(grad2[0], grad3[3]))
        grad1, grad2 = run("7", "8", "normal3", "adam")
        grad3 = run("9", "9", "aggregate3", "adam")
        self.assertTrue(np.allclose(grad1[1], grad3[1]))
        self.assertTrue(np.allclose(grad1[2] + grad2[1], grad3[2]))
        self.assertTrue(np.allclose(grad1[0] + grad2[2], grad3[0]))
        self.assertTrue(np.allclose(grad2[0], grad3[3]))
        grad1, grad2 = run("10", "11", "normal4", "adagrad")
        grad3 = run("12", "12", "aggregate4", "adagrad")
        self.assertTrue(np.allclose(grad1[1], grad3[1]))
        self.assertTrue(np.allclose(grad1[2] + grad2[1], grad3[2]))
        self.assertTrue(np.allclose(grad1[0] + grad2[2], grad3[0]))
        self.assertTrue(np.allclose(grad2[0], grad3[3]))
        grad1, grad2 = run("13", "14", "normal5", "rmsprop")
        grad3 = run("15", "15", "aggregate5", "rmsprop")
        self.assertTrue(np.allclose(grad1[1], grad3[1]))
        self.assertTrue(np.allclose(grad1[2] + grad2[1], grad3[2]))
        self.assertTrue(np.allclose(grad1[0] + grad2[2], grad3[0]))
        self.assertTrue(np.allclose(grad2[0], grad3[3]))

def suite():
  return unittest.TestLoader().loadTestsFromTestCase(GradientAggregationTest)

if __name__ == '__main__':
  unittest.TextTestRunner().run(suite())

