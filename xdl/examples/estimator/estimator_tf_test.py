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
import tensorflow as tf
import time

''' 
local run:
  python estimator_tf_test.py --run_mode=local --ckpt_dir=./ckpt --task_type=train|eval|predict|train_and_eval 
'''

def input_fn():
    dense = xdl.mock_dense_op(shape=[1, 16], value=0.01)
    labels = xdl.mock_dense_op(shape=[1, 1], value=1.0) 
    ids, values, segments = xdl.mock_sparse_op(dense_shape=[1, 16])
    sparse = xdl.SparseTensor(ids, values, segments)
    emb = xdl.embedding("sparse", sparse, xdl.Ones(), 1, 16, 'sum')
    dense.set_shape([None, 16])
    labels.set_shape([None, 1])
    return [dense, emb], labels

def eval_input_fn():
    dense = xdl.mock_dense_op(shape=[1, 16], value=0.01)
    labels = xdl.mock_dense_op(shape=[1, 1], value=1.0) 
    ids, values, segments = xdl.mock_sparse_op(dense_shape=[1, 16])
    sparse = xdl.SparseTensor(ids, values, segments)
    emb = xdl.embedding("sparse", sparse, xdl.Ones(), 1, 16, 'sum')
    dense.set_shape([None, 16])
    labels.set_shape([None, 1])
    return [dense, emb], labels

@xdl.tf_wrapper(is_training=True)
def model_fn(inputs, labels):
    dense = inputs[0]
    emb = inputs[1]
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        fc2 = fc(dense, [16, 1], [1])
        logits = fc2 + emb
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean'), logits

def fc(inputs, w_shape, b_shape):
    w = tf.get_variable(
        "weights", 
        w_shape, 
        initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.36), 
        regularizer=tf.nn.l2_loss)
    b = tf.get_variable(
        "bias", 
        b_shape, 
        initializer=tf.uniform_unit_scaling_initializer(factor=0.1, seed=10, dtype=tf.float32))
    return tf.matmul(inputs, w)

def train():
    estimator = xdl.Estimator(
        model_fn=model_fn, 
        optimizer=xdl.SGD(0.5))
    estimator.train(input_fn, max_step=2000, checkpoint_interval=1000)

def evaluate():
    estimator = xdl.Estimator(
        model_fn=model_fn, 
        optimizer=xdl.SGD(0.5))
    estimator.evaluate(input_fn, checkpoint_version="", max_step=2000)

def predict():
    estimator = xdl.Estimator(
        model_fn=model_fn, 
        optimizer=xdl.SGD(0.5))
    estimator.predict(input_fn, checkpoint_version="", max_step=2000)

def train_and_evaluate():
    estimator = xdl.Estimator(
        model_fn=model_fn, 
        optimizer=xdl.SGD(0.5))
    
    estimator.train_and_evaluate(train_input_fn=input_fn,
                                 eval_input_fn=eval_input_fn,
                                 eval_interval=1000,
                                 eval_steps=200,
                                 checkpoint_interval=1000,
                                 max_step=5000)

task_type = xdl.get_task_type()
if task_type == None:
    task_type = 'train'

if task_type == 'train':
    train()    
elif task_type == 'eval':
    evaluate()
elif task_type == 'predict':
    predict()
elif task_type == 'train_and_eval':
    train_and_evaluate()
else:
    raise Exception("unsupported type")
    
