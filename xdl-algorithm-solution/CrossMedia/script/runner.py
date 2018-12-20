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

import os
import xdl
import tensorflow as tf
import numpy as np
from xdl.python.backend.tf.tf_backend import ams_gear, ams_main

main_input_length = 16 * 2 + 4 * 2 + 12 * 2
img_data_source = "file://" + os.path.dirname(os.path.realpath(__file__)) + '/imgs/data'

def fc(inputs, x, y, name, activation = tf.nn.relu):
  with tf.variable_scope(name):
    w = tf.get_variable("weights",
                        [x, y],
                        initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.1 / x),
                        regularizer=tf.nn.l2_loss)
    b = tf.get_variable("bias",
                        [y],
                        initializer=tf.constant_initializer(0.005 / x, dtype=tf.float32))
    if activation is not None:
      return activation(tf.matmul(inputs, w) + b)
    else:
      return tf.matmul(inputs, w) + b

def reader():
  path = os.path.dirname(os.path.realpath(__file__)) + '/data.txt'
  data_io = xdl.DataIO("tdm", file_type=xdl.parsers.txt, fs_type=xdl.fs.local)
  data_io.epochs(100)
  data_io.threads(1)
  data_io.batch_size(16)
  data_io.label_count(1)
  data_io.feature(name='user0', type=xdl.features.sparse)
  data_io.feature(name='user1', type=xdl.features.sparse)
  data_io.feature(name='ad0', type=xdl.features.dense, nvec=4)
  data_io.feature(name='ad1', type=xdl.features.dense, nvec=4)
  data_io.feature(name='user_img', type=xdl.features.sparse, serialized=True)
  data_io.feature(name='ad_img', type=xdl.features.sparse, serialized=True)
  data_io.add_path(path)
  data_io.startup()
  return data_io

def main_model(u0, u1, ad0, ad1, label, ids0, ids1, gear_inputs):
  graph0 = tf.sparse_segment_sum(gear_inputs[0], tf.range(tf.shape(gear_inputs[0])[0]), ids0, num_segments=tf.shape(label)[0])
  graph1 = tf.sparse_segment_sum(gear_inputs[1], tf.range(tf.shape(gear_inputs[1])[0]), ids1, num_segments=tf.shape(label)[0])
  input_layer = tf.concat([u0, u1, ad0, ad1, graph0, graph1], -1)
  l1 = fc(input_layer, main_input_length, 100, "mainfc1")
  l2 = fc(l1, 100, 10, "mainfc2")
  l3 = fc(l2, 10, 1, "mainfc3")
  return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=l3, labels=label))

def user_graph_model(imgs):
  l1 = fc(imgs, 4096, 256, "user_graphfc1")
  l2 = fc(l1, 256, 64, "user_graphfc2")
  l3 = fc(l2, 64, 12, "user_graphfc3")
  return l3

def user_graph_train(forward, backward, grads):
  server_size = xdl.current_env().model_server_size(xdl.current_env().task_name())
  server_id = xdl.current_env().task_id()
  forward_img = xdl.hdfs_data_source_op(forward, img_data_source, server_size, server_id, 4096, xdl.DataType.float)
  backward_img = xdl.hdfs_data_source_op(backward, img_data_source, server_size, server_id, 4096, xdl.DataType.float)
  forward_img.set_shape([None, 4096])
  backward_img.set_shape([None, 4096])
  grads.set_shape([None, 12])
  predict = ams_gear([forward_img], [backward_img], grads)(user_graph_model)(None)[0]
  with xdl.model_scope("ams_gear_backward"):
    optimizer = xdl.Adam(0.0005).optimize(update_global_step=False)
  return predict, optimizer

def ad_graph_model(imgs):
  l1 = fc(imgs, 4096, 256, "ad_graphfc1")
  l2 = fc(l1, 256, 64, "ad_graphfc2")
  l3 = fc(l2, 64, 12, "ad_graphfc3")
  return l3

def ad_graph_train(forward, backward, grads):
  server_size = xdl.current_env().model_server_size(xdl.current_env().task_name())
  server_id = xdl.current_env().task_id()
  forward_img = xdl.hdfs_data_source_op(forward, img_data_source, server_size, server_id, 4096, xdl.DataType.float)
  backward_img = xdl.hdfs_data_source_op(backward, img_data_source, server_size, server_id, 4096, xdl.DataType.float)
  forward_img.set_shape([None, 4096])
  backward_img.set_shape([None, 4096])
  grads.set_shape([None, 12])
  predict = ams_gear([forward_img], [backward_img], grads)(ad_graph_model)(None)[0]
  with xdl.model_scope("ams_gear_backward"):
    optimizer = xdl.Adam(0.0005).optimize(update_global_step=False)
  return predict, optimizer

def to_tf_segment_id(seg):
  l = []
  segx = seg.tolist()
  z = 0
  for i in range(len(segx)):
    l += range(z, segx[i])
    z = segx[i]
  return np.array(l, np.int32)

def run():
  user_ms = xdl.ModelServer(
      "user_graph", user_graph_train, xdl.DataType.float,
      xdl.ModelServer.Forward.UniqueCache(xdl.get_task_num()),
      xdl.ModelServer.Backward.UniqueCache(xdl.get_task_num()))
  xdl.current_env().start_model_server(user_ms)
  ad_ms = xdl.ModelServer(
      "ad_graph", ad_graph_train, xdl.DataType.float,
      xdl.ModelServer.Forward.UniqueCache(xdl.get_task_num()),
      xdl.ModelServer.Backward.UniqueCache(xdl.get_task_num()))
  xdl.current_env().start_model_server(ad_ms)
  batch = reader().read()

  user0 = xdl.embedding("user0", batch["user0"], xdl.TruncatedNormal(stddev=0.001),
                        16, 2 * 1024 * 1024, "sum", vtype="hash")

  user1 = xdl.embedding("user1", batch["user1"], xdl.TruncatedNormal(stddev=0.001),
                        16, 2 * 1024 * 1024, "sum", vtype="hash")

  ad0 = batch["ad0"]
  ad1 = batch["ad1"]
  img0 = user_ms(batch["user_img"].ids)
  ids0 = xdl.py_func(to_tf_segment_id, [batch["user_img"].segments], [np.int32])[0]
  img1 = ad_ms(batch["ad_img"].ids)
  ids1 = xdl.py_func(to_tf_segment_id, [batch["ad_img"].segments], [np.int32])[0]
  label = batch['label']
  loss = ams_main(main_model)(user0, user1, ad0, ad1, label, ids0, ids1, gear_inputs=[img0, img1])

  optimizer = xdl.Adam(0.0005).optimize()

  run_ops = [loss, optimizer]

  sess = xdl.TrainSession([])
  while not sess.should_stop():
    values = sess.run(run_ops)
    if values is not None:
      print 'loss: ', values[0]

run()
