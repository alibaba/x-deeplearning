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
import numpy as np
from data_iterator import DataIterator
from xdl.python.lib.error import OutOfRange

class SampleIO(object):
    def __init__(self,
                 train_file="local_train_splitByUser",
                 test_file="local_test_splitByUser",
                 uid_voc="uid_voc.pkl",
                 mid_voc="mid_voc.pkl",
                 cat_voc="cat_voc.pkl",
                 item_info='item-info',
                 reviews_info='reviews-info',
                 batch_size=128,
                 maxlen=100,
                 embedding_dim=None,
                 return_neg=True):
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.return_neg = return_neg
        self.train_data = DataIterator(
            train_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info, batch_size, maxlen, shuffle_each_epoch=False)
        self.test_data = DataIterator(
            test_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info, batch_size, maxlen)
        self.n_uid, self.n_mid, self.n_cat = self.train_data.get_n()

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat

    def next_train(self):
        if self.return_neg:
            return self._py_func(self._next_train)
        else:
            return self._py_func(self._next_train, sparse_cnt=5)

    def next_test(self):
        if self.return_neg:
            return self._py_func(self._next_test)
        else:
            return self._py_func(self._next_test, sparse_cnt=5)

    def _next_train(self):
        try:
            src, tgt = self.train_data.next()
        except StopIteration:
            self.src = self.tgt = None
            raise OutOfRange("train end")
        return self.prepare_data(src, tgt, self.maxlen, return_neg=self.return_neg)

    def _next_test(self):
        try:
            src, tgt = self.test_data.next()
        except StopIteration:
            self.src = self.tgt = None
            raise OutOfRange("test end")
        return self.prepare_data(src, tgt, self.maxlen, return_neg=self.return_neg)

    def _py_func(self, fn, sparse_cnt=7):
        types = []
        for _ in range(sparse_cnt):
            types.extend([np.int64, np.float32, np.int32])
        types.extend([np.float32, np.float32, np.int32])
        types.extend([np.int32 for _ in range(5)])
        datas = xdl.py_func(fn, [], output_type=types)
        sparse_tensors = []
        for i in range(sparse_cnt):
            sparse_tensors.append(xdl.SparseTensor(
                    datas[3 * i], datas[3 * i + 1], datas[3 * i + 2]))
        return sparse_tensors + datas[sparse_cnt * 3:]

    def prepare_data(self, input, target, maxlen=None, return_neg=False):
        # x: a list of sentences
        lengths_x = [len(s[4]) for s in input]
        seqs_mid = [inp[3] for inp in input]
        seqs_cat = [inp[4] for inp in input]
        noclk_seqs_mid = [inp[5] for inp in input]
        noclk_seqs_cat = [inp[6] for inp in input]

        if maxlen is not None:
            new_seqs_mid = []
            new_seqs_cat = []
            new_noclk_seqs_mid = []
            new_noclk_seqs_cat = []
            new_lengths_x = []
            for l_x, inp in zip(lengths_x, input):
                if l_x > maxlen:
                    new_seqs_mid.append(inp[3][l_x - maxlen:])
                    new_seqs_cat.append(inp[4][l_x - maxlen:])
                    new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                    new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                    new_lengths_x.append(maxlen)
                else:
                    new_seqs_mid.append(inp[3])
                    new_seqs_cat.append(inp[4])
                    new_noclk_seqs_mid.append(inp[5])
                    new_noclk_seqs_cat.append(inp[6])
                    new_lengths_x.append(l_x)
            lengths_x = new_lengths_x
            seqs_mid = new_seqs_mid
            seqs_cat = new_seqs_cat
            noclk_seqs_mid = new_noclk_seqs_mid
            noclk_seqs_cat = new_noclk_seqs_cat

            if len(lengths_x) < 1:
                return None, None, None, None

        n_samples = len(seqs_mid)
        maxlen_x = np.max(lengths_x) + 1
        neg_samples = len(noclk_seqs_mid[0][0])

        mid_his = np.zeros((n_samples, maxlen_x)).astype('int64')
        cat_his = np.zeros((n_samples, maxlen_x)).astype('int64')
        noclk_mid_his = np.zeros(
            (n_samples, maxlen_x, neg_samples)).astype('int64')
        noclk_cat_his = np.zeros(
            (n_samples, maxlen_x, neg_samples)).astype('int64')
        mid_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
        for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
            mid_mask[idx, :lengths_x[idx] + 1] = 1.
            mid_his[idx, :lengths_x[idx]] = s_x
            cat_his[idx, :lengths_x[idx]] = s_y
            noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
            noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

        uids = np.array([inp[0] for inp in input], dtype=np.int64)
        mids = np.array([inp[1] for inp in input], dtype=np.int64)
        cats = np.array([inp[2] for inp in input], dtype=np.int64)

        id_values = np.ones([n_samples], np.float32)
        his_values = np.ones([n_samples * maxlen_x], np.float32)
        neg_his_values = np.ones(
            [n_samples * maxlen_x * neg_samples], np.float32)

        id_seg = np.array([i + 1 for i in range(n_samples)], dtype=np.int32)
        his_seg = np.array(
            [i + 1 for i in range(n_samples * maxlen_x)], dtype=np.int32)
        neg_his_seg = np.array(
            [i + 1 for i in range(n_samples * maxlen_x * neg_samples)], dtype=np.int32)

        results = []
        for e in [uids, mids, cats]:
            results.append(np.reshape(e, (-1)))
            results.append(id_values)
            results.append(id_seg)
        for e in [mid_his, cat_his]:
            results.append(np.reshape(e, (-1)))
            results.append(his_values)
            results.append(his_seg)
        if return_neg:
            for e in [noclk_mid_his, noclk_cat_his]:
                results.append(np.reshape(e, (-1)))
                results.append(neg_his_values)
                results.append(neg_his_seg)
        results.extend(
            [mid_mask, np.array(target, dtype=np.float32), np.array(lengths_x, dtype=np.int32)])
        # for split
        results.append(np.array([n_samples, n_samples], dtype=np.int32))
        # shape
        results.extend([np.array([-1, self.embedding_dim], dtype=np.int32),
                        np.array([-1, maxlen_x, self.embedding_dim],
                                 dtype=np.int32),
                        np.array(
                            [-1, maxlen_x, neg_samples, self.embedding_dim], dtype=np.int32),
                        np.array([-1, maxlen_x], dtype=np.int32)])
        return results
