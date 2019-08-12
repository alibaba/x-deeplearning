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
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from xdl.python.utils.collections import reset_collections
from xdl.python.training.gradient_utils import reset_gradients

data_dir = './data'
mnist_data = input_data.read_data_sets(data_dir)


def read_train(batch_size=100):
    global mnist_data
    images, labels = mnist_data.train.next_batch(batch_size)
    labels = np.asarray(labels, np.float32)
    return images, labels


def read_test():
    global mnist_data
    labels = np.asarray(mnist_data.test.labels, np.float32)
    return mnist_data.test.images, labels


def train():
    images, labels = xdl.py_func(read_train, [], output_type=[
                                 np.float32, np.float32])
    images_test, labels_test = xdl.py_func(
        read_test, [], output_type=[np.float32, np.float32])
    with xdl.model_scope('train'):
      loss = model(images, labels)
      train_op = xdl.Adagrad(0.5).optimize()
      train_sess = xdl.TrainSession()

    with xdl.model_scope('test'):
      accuracy = eval_model(images_test, labels_test)
      eval_sess = xdl.TrainSession()
    for _ in range(100):
        for _ in range(1000):
            train_sess.run(train_op)

        print("accuracy %s" % eval_sess.run(accuracy))


def fc(inputs, w_shape, b_shape):
    weight = tf.get_variable("weights",
                             w_shape,
                             initializer=tf.zeros_initializer(tf.float32))
    bias = tf.get_variable("bias",
                           b_shape,
                           initializer=tf.zeros_initializer(tf.float32))
    return tf.matmul(inputs, weight) + bias


@xdl.tf_wrapper(is_training=True)
def model(images, labels):
    with tf.variable_scope("train"):
        y = fc(images, [784, 10], [10])
        labels = tf.cast(labels, tf.int64)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=y)
    return loss


@xdl.tf_wrapper(is_training=False)
def eval_model(images, labels):
    with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
        eval_y = fc(images, [784, 10], [10])
        labels_test = tf.cast(labels, tf.int64)
        correct_prediction = tf.equal(tf.argmax(eval_y, 1), labels_test)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


train()
