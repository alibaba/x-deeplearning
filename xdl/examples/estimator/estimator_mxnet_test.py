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

import mxnet as mx
import xdl
import time

''' 
local run:
  python estimator_mxnet_test.py --run_mode=local --ckpt_dir=./ckpt --task_type=train|eval|predict|train_and_eval 
'''
def input_fn():
    dense = xdl.mock_dense_op(shape=[1, 16], value=0.01)
    indicator = xdl.mock_dense_op(shape=[5], value=0.0)
    labels = xdl.mock_dense_op(shape=[5, 1], value=1.0) 
    ids, values, segments = xdl.mock_sparse_op(dense_shape=[1, 16])
    sparse = xdl.SparseTensor(ids, values, segments)
    sparse.set_shape([1,16])
    emb = xdl.embedding("sparse", sparse, xdl.Ones(), 1, 16, 'sum')
    dense.set_shape([1, 16])
    indicator.set_shape([5])
    labels.set_shape([5, 1])
    return [dense, emb, indicator], labels

def eval_input_fn():
    dense = xdl.mock_dense_op(shape=[1, 16], value=0.01)
    indicator = xdl.mock_dense_op(shape=[5], value=0.0)
    labels = xdl.mock_dense_op(shape=[5, 1], value=1.0) 
    ids, values, segments = xdl.mock_sparse_op(dense_shape=[1, 16])
    sparse = xdl.SparseTensor(ids, values, segments)
    sparse.set_shape([1,16])
    emb = xdl.embedding("sparse", sparse, xdl.Ones(), 1, 16, 'sum')
    dense.set_shape([1, 16])
    indicator.set_shape([5])
    labels.set_shape([5, 1])
    return [dense, emb, indicator], labels

@xdl.mxnet_wrapper(is_training=True)
def model_fn(inputs, label):
    din = mx.symbol.concat(inputs[0], inputs[1])
    din = mx.sym.take(din, inputs[2])
    dout = mx.sym.FullyConnected(data=din, num_hidden=1, name='fc1')
    prop = mx.symbol.SoftmaxOutput(data=dout, label=label, grad_scale=(1.0 / 4))
    loss = - mx.symbol.sum(mx.symbol.log(prop) * label) / 4
    return loss, prop

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
    
