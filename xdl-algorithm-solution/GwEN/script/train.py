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
import sys
import time
import math
import random

import tensorflow as tf
import numpy

from model import *
from utils import *
from sample_io import SampleIO

import xdl
from xdl.python.training.train_session import QpsMetricsHook, MetricsPrinterHook

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0

def get_data_prefix():
    return xdl.get_config('data_dir')

train_file = os.path.join(get_data_prefix(), "local_train_splitByUser")
test_file = os.path.join(get_data_prefix(), "local_test_splitByUser")
uid_voc = os.path.join(get_data_prefix(), "uid_voc.pkl")
mid_voc = os.path.join(get_data_prefix(), "mid_voc.pkl")
cat_voc = os.path.join(get_data_prefix(), "cat_voc.pkl")
item_info = os.path.join(get_data_prefix(), 'item-info')
reviews_info = os.path.join(get_data_prefix(), 'reviews-info')

def train(train_file=train_file,
          test_file=test_file,
          uid_voc=uid_voc,
          mid_voc=mid_voc,
          cat_voc=cat_voc,
          item_info=item_info,
          reviews_info=reviews_info,
          batch_size=128,
          maxlen=100,
          test_iter=700):
    if xdl.get_config('model') == 'din':
        model = Model_DIN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif xdl.get_config('model') == 'gwen':
        model = Model_GwEN(
            EMBEDDING_DIM, HIDDEN_SIZE)
    elif xdl.get_config('model') == 'dien':
        model = Model_DIEN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    else:
        raise Exception('only support din and dien')

    sample_io = SampleIO(train_file, test_file, uid_voc, mid_voc,
                         cat_voc, item_info, reviews_info,
                         batch_size, maxlen, EMBEDDING_DIM)
    with xdl.model_scope('train'):
        train_ops = model.build_final_net(EMBEDDING_DIM, sample_io)
        lr = 0.001
        # Adam Adagrad
        train_ops.append(xdl.Adam(lr).optimize())
        hooks = []
        log_format = "[%(time)s] lstep[%(lstep)s] gstep[%(gstep)s] lqps[%(lqps)s] gqps[%(gqps)s] loss[%(loss)s]"
        hooks = [QpsMetricsHook(), MetricsPrinterHook(log_format)]
        if xdl.get_task_index() == 0:
            hooks.append(xdl.CheckpointHook(xdl.get_config('checkpoint', 'save_interval')))
        train_sess = xdl.TrainSession(hooks=hooks)

    with xdl.model_scope('test'):
        test_ops = model.build_final_net(
            EMBEDDING_DIM, sample_io, is_train=False)
        test_sess = xdl.TrainSession()

    model.run(train_ops, train_sess, test_ops, test_sess, test_iter=test_iter)

def test(train_file=train_file,
         test_file=test_file,
         uid_voc=uid_voc,
         mid_voc=mid_voc,
         cat_voc=cat_voc,
         batch_size=128,
         maxlen=100):
   # sample_io
    sample_io = SampleIO(train_file, test_file, uid_voc, mid_voc,
                         cat_voc, batch_size, maxlen, EMBEDDING_DIM)

    if xdl.get_config('model') == 'din':    
        model = Model_DIN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    if xdl.get_config('model') == 'gwen':    
        model = Model_GwEN(
            EMBEDDING_DIM, HIDDEN_SIZE)
    elif xdl.get_config('model') == 'dien':    
        model = Model_DIEN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    else:
        raise Exception('only support din and dien model')

    # test
    datas = sample_io.next_test()
    test_ops = tf_test_model(
        *model.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))
    eval_sess = xdl.TrainSession()
    print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' %
          eval_model(eval_sess, test_ops))

if __name__ == '__main__':
    SEED = xdl.get_config("seed")
    if SEED is None:
        SEED = 3
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

    job_type = xdl.get_config("job_type")
    if job_type == 'train':
        train()
    elif job_type == 'test':
        test()
    else:
        print('job type must be train or test, do nothing...')
