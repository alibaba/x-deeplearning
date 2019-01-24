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

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
import xdl
import time
import sys
import random
import math
import numpy
import datetime
from xdl.python.utils.metrics import add_metrics

best_auc = 0.0

def eval_model(sess, test_ops):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    while not sess.should_stop():
        nums += 1
        values = sess.run(test_ops)
        if values is None:
            break
        prob, loss, acc, aux_loss, target = values
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])

    sess._finish = False
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum

class DataTensors(object):
    def __init__(self, datas, embedding_dim):
        self.embedding_dim = embedding_dim
        self.uid = tf.reshape(datas[0], datas[-4])
        self.uid.set_shape([None, self.embedding_dim])
        self.mid = tf.reshape(datas[1], datas[-4])
        self.mid.set_shape([None, self.embedding_dim])
        self.cat = tf.reshape(datas[2], datas[-4])
        self.cat.set_shape([None, self.embedding_dim])
        self.mid_list = tf.reshape(datas[3], datas[-3])
        self.mid_list.set_shape([None, None, self.embedding_dim])
        self.cat_list = tf.reshape(datas[4], datas[-3])
        self.cat_list.set_shape([None, None, self.embedding_dim])
        self.mid_neg_list = tf.reshape(datas[5], datas[-2])
        self.mid_neg_list.set_shape([None, None, None, self.embedding_dim])
        self.cat_neg_list = tf.reshape(datas[6], datas[-2])
        self.cat_neg_list.set_shape([None, None, None, self.embedding_dim])
        self.mask = tf.reshape(datas[7], datas[-1])
        self.mask.set_shape([None, None])
        self.target = tf.reshape(datas[8], [-1, 2])
        self.target.set_shape([None, 2])
        self.seq_len = tf.reshape(datas[9], [-1])
        self.seq_len.set_shape([None])

class Model(object):
    def __init__(self, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        self.embedding_dim = EMBEDDING_DIM
        self.hidden_size = HIDDEN_SIZE
        self.attention_size = ATTENTION_SIZE
        self.use_negsampling = use_negsampling

    def build_tf_net(self, datas, is_train=True):
        self.is_train = is_train
        self.tensors = DataTensors(datas, self.embedding_dim)
        self.item_eb = tf.concat([self.tensors.mid, self.tensors.cat], 1)
        self.item_his_eb = tf.concat(
            [self.tensors.mid_list, self.tensors.cat_list], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.tensors.mid_neg_list[:, :, 0, :], self.tensors.cat_neg_list[:, :, 0, :]], -1)
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                [-1, tf.shape(self.tensors.mid_neg_list)[1], 2 * self.embedding_dim])

            self.noclk_his_eb = tf.concat(
                [self.tensors.mid_neg_list, self.tensors.cat_neg_list], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

    def build_fcn_net(self, inp, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=True)
        dnn1 = tf.layers.dense(
            bn1, 200, activation=None, kernel_initializer=get_tf_initializer(), name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, scope='prelu_1')
        dnn2 = tf.layers.dense(
            dnn1, 80, activation=None, kernel_initializer=get_tf_initializer(), name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, scope='prelu_2')
        dnn3 = tf.layers.dense(
            dnn2, 2, activation=None, kernel_initializer=get_tf_initializer(), name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            ctr_loss = - \
                tf.reduce_mean(tf.log(self.y_hat) * self.tensors.target)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            else:
                self.aux_loss = tf.constant(0.0)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.round(self.y_hat), self.tensors.target), tf.float32))

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_),
                                   [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - \
            tf.reshape(tf.log(1.0 - noclick_prop_),
                       [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net', reuse=tf.AUTO_REUSE):
        with tf.variable_scope("aux", reuse=reuse):
            bn1 = tf.layers.batch_normalization(
                inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE, training=True)
            #, kernel_initializer=tf.constant_initializer(0.3)
            dnn1 = tf.layers.dense(bn1, 100, activation=None, kernel_initializer=get_tf_initializer(),
                                   name='f1' + stag, reuse=tf.AUTO_REUSE)
            dnn1 = tf.nn.sigmoid(dnn1)
            dnn2 = tf.layers.dense(dnn1, 50, activation=None, kernel_initializer=get_tf_initializer(),
                                   name='f2' + stag, reuse=tf.AUTO_REUSE)
            dnn2 = tf.nn.sigmoid(dnn2)
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, kernel_initializer=get_tf_initializer(),
                                   name='f3' + stag, reuse=tf.AUTO_REUSE)
            y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def build_final_net(self, EMBEDDING_DIM, sample_io, is_train=True):
        @xdl.tf_wrapper(is_training=True)
        def tf_train_model(*inputs):
            with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
                self.build_tf_net(inputs, is_train)
            train_ops = self.train_ops()
            return train_ops[0], train_ops[1:]

        @xdl.tf_wrapper(is_training=False)
        def tf_test_model(*inputs):
            with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
                self.build_tf_net(inputs, is_train)
            test_ops = self.test_ops()
            return test_ops[0], test_ops[1:]

        if is_train:
            datas = sample_io.next_train()
            train_ops = tf_train_model(
                *self.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))
            return train_ops
        else:
            datas = sample_io.next_test()
            test_ops = tf_test_model(
                *self.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))
            return test_ops

    def train_ops(self):
        return [self.loss, self.accuracy, self.aux_loss]

    def test_ops(self):
        return [self.y_hat, self.loss, self.accuracy, self.aux_loss, self.tensors.target]

    def run_test(self, test_ops, test_sess):
        if xdl.get_task_index() == 0 and test_ops is not None and test_sess is not None:
            print('test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' %
                  eval_model(test_sess, test_ops))

    def run(self, train_ops, train_sess, test_iter=100):
        step = 0
        for epoch in range(1):
            while not train_sess.should_stop():
                values = train_sess.run(train_ops)
                if values is None:
                    break
                loss, acc, aux_loss = values
                step += 1
                if (step % test_iter) == 0:
                    print('step: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- tran_aux_loss: %.4f' %
                          (step, loss, acc, aux_loss))
            train_sess._finish = False

class Model_DIEN(Model):
    def __init__(self, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DIEN, self).__init__(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)

    def build_tf_net(self, datas, is_train=True):

        super(Model_DIEN,
              self).build_tf_net(datas, is_train)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.hidden_size, kernel_initializer=get_tf_initializer()), inputs=self.item_his_eb, sequence_length=self.tensors.seq_len, dtype=tf.float32, scope="gru1")

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.noclk_item_his_eb[:, 1:, :],
                                         self.tensors.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, self.attention_size, self.tensors.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(self.hidden_size, kernel_initializer=get_tf_initializer()), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(
                                                         alphas, -1),
                                                     sequence_length=self.tensors.seq_len, dtype=tf.float32,
                                                     scope="gru2")

        inp = tf.concat([self.tensors.uid, self.item_eb,
                         final_state2, self.item_his_eb_sum], 1)

        self.build_fcn_net(inp, use_dice=True)

    def xdl_embedding(self, datas, EMBEDDING_DIM, n_uid, n_mid, n_cat):
        results = []
        uid_emb = xdl.embedding("uid_embedding", datas[0], get_xdl_initializer(),
                                EMBEDDING_DIM, n_uid, 'sum')
        results.append(uid_emb)
        for i in range(3):
            mid_emb = xdl.embedding("mid_embedding", datas[i * 2 + 1], get_xdl_initializer(),
                                    EMBEDDING_DIM, n_mid, 'sum')
            cat_emb = xdl.embedding("cat_embedding", datas[2 * (i + 1)], get_xdl_initializer(),
                                    EMBEDDING_DIM, n_cat, 'sum')
            results.append(mid_emb)
            results.append(cat_emb)
        return results + datas[7:]

    def build_final_net(self, EMBEDDING_DIM, sample_io, is_train=True):
        @xdl.tf_wrapper(is_training=True)
        def tf_train_model(*inputs):
            with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
                self.build_tf_net(inputs, is_train)
            train_ops = self.train_ops()
            return train_ops[0], train_ops[1:]

        @xdl.tf_wrapper(is_training=False)
        def tf_test_model(*inputs):
            with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
                self.build_tf_net(inputs, is_train)
            test_ops = self.test_ops()
            return test_ops[0], test_ops[1:]
        if is_train:
            datas = sample_io.next_train()
            train_ops = tf_train_model(
                *self.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))
            return train_ops
        else:
            datas = sample_io.next_test()
            test_ops = tf_test_model(
                *self.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))
            return test_ops

    def run(self, train_ops, train_sess, test_ops=None, test_sess=None, test_iter=100, save_iter=1500):
        iter = 0
        for epoch in range(1):
            while not train_sess.should_stop():
                values = train_sess.run(train_ops)
                if values is None:
                    break
                loss, acc, aux_loss, _ = values
                add_metrics("loss", loss)
                add_metrics("time", datetime.datetime.now(
                ).strftime('%Y-%m-%d %H:%M:%S'))

                iter += 1
                if (iter % test_iter) == 0:
                    self.run_test(test_ops, test_sess)
            train_sess._finish = False


class Model_DIN(Model):
    def __init__(self, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN, self).__init__(EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling)

    def build_tf_net(self, datas, is_train=True):
        super(Model_DIN,
              self).build_tf_net(datas, is_train)
        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(
                self.item_eb, self.item_his_eb, self.attention_size, self.tensors.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
        inp = tf.concat([self.tensors.uid, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, att_fea], -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)

    def xdl_embedding(self, datas, EMBEDDING_DIM, n_uid, n_mid, n_cat):
        results = []
        uid_emb = xdl.embedding("uid_embedding", datas[0], get_xdl_initializer(),
                                EMBEDDING_DIM, n_uid, 'sum')
        results.append(uid_emb)
        for i in range(3):
            mid_emb = xdl.embedding("mid_embedding", datas[i * 2 + 1], get_xdl_initializer(),
                                    EMBEDDING_DIM, n_mid, 'sum')
            cat_emb = xdl.embedding("cat_embedding", datas[2 * (i + 1)], get_xdl_initializer(),
                                    EMBEDDING_DIM, n_cat, 'sum')
            results.append(mid_emb)
            results.append(cat_emb)
        return results + datas[7:]

    def build_final_net(self, EMBEDDING_DIM, sample_io, is_train=True):
        @xdl.tf_wrapper(is_training=True)
        def tf_train_model(*inputs):
            with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
                self.build_tf_net(inputs, is_train)
            train_ops = self.train_ops()
            return train_ops[0], train_ops[1:]

        @xdl.tf_wrapper(is_training=False)
        def tf_test_model(*inputs):
            with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
                self.build_tf_net(inputs, is_train)
            test_ops = self.test_ops()
            return test_ops[0], test_ops[1:]
        if is_train:
            datas = sample_io.next_train()
            train_ops = tf_train_model(
                *self.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))
            return train_ops
        else:
            datas = sample_io.next_test()
            test_ops = tf_test_model(
                *self.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))
            return test_ops

    def run(self, train_ops, train_sess, test_ops=None, test_sess=None, test_iter=100):
        iter = 0
        for epoch in range(1):
            while not train_sess.should_stop():
                values = train_sess.run(train_ops)
                if values is None:
                    break
                loss, acc, aux_loss, _ = values
                add_metrics("loss", loss)
                add_metrics("time", datetime.datetime.now(
                ).strftime('%Y-%m-%d %H:%M:%S'))

                iter += 1
                if (iter % test_iter) == 0:
                    self.run_test(test_ops, test_sess)
            train_sess._finish = False



class Model_GwEN(Model):
    def __init__(self, EMBEDDING_DIM, HIDDEN_SIZE, use_negsampling=False):
        super(Model_GwEN, self).__init__(EMBEDDING_DIM, HIDDEN_SIZE,
                                        use_negsampling)

    def build_tf_net(self, datas, is_train=True):
        super(Model_GwEN,
              self).build_tf_net(datas, is_train)
        # Attention layer
        inp = tf.concat([self.tensors.uid, self.item_eb, self.item_his_eb_sum], -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=False)

    def xdl_embedding(self, datas, EMBEDDING_DIM, n_uid, n_mid, n_cat):
        results = []
        uid_emb = xdl.embedding("uid_embedding", datas[0], get_xdl_initializer(),
                                EMBEDDING_DIM, n_uid, 'sum')
        results.append(uid_emb)
        for i in range(3):
            mid_emb = xdl.embedding("mid_embedding", datas[i * 2 + 1], get_xdl_initializer(),
                                    EMBEDDING_DIM, n_mid, 'sum')
            cat_emb = xdl.embedding("cat_embedding", datas[2 * (i + 1)], get_xdl_initializer(),
                                    EMBEDDING_DIM, n_cat, 'sum')
            results.append(mid_emb)
            results.append(cat_emb)
        return results + datas[7:]

    def build_final_net(self, EMBEDDING_DIM, sample_io, is_train=True):
        @xdl.tf_wrapper(is_training=True)
        def tf_train_model(*inputs):
            with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
                self.build_tf_net(inputs, is_train)
            train_ops = self.train_ops()
            return train_ops[0], train_ops[1:]

        @xdl.tf_wrapper(is_training=False)
        def tf_test_model(*inputs):
            with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
                self.build_tf_net(inputs, is_train)
            test_ops = self.test_ops()
            return test_ops[0], test_ops[1:]
        if is_train:
            datas = sample_io.next_train()
            train_ops = tf_train_model(
                *self.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))
            return train_ops
        else:
            datas = sample_io.next_test()
            test_ops = tf_test_model(
                *self.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))
            return test_ops

    def run(self, train_ops, train_sess, test_ops=None, test_sess=None, test_iter=100):
        iter = 0
        for epoch in range(1):
            while not train_sess.should_stop():
                values = train_sess.run(train_ops)
                if values is None:
                    break
                loss, acc, aux_loss, _ = values
                add_metrics("loss", loss)
                add_metrics("time", datetime.datetime.now(
                ).strftime('%Y-%m-%d %H:%M:%S'))

                iter += 1
                if (iter % test_iter) == 0:
                    self.run_test(test_ops, test_sess)
            train_sess._finish = False



