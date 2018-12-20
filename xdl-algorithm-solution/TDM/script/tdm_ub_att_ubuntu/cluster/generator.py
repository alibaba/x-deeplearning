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

#!/usr/bin/env python

from __future__ import print_function

import sys
import os

import time
import argparse
import json
import random
import multiprocessing as mp

import numpy as np

import tree_builder


class Generator:
    def __init__(self,
                 train_data_file,
                 test_data_file,
                 tree_pb_file,
                 train_sample_file,
                 test_sample_file,
                 train_sample_seg_cnt,
                 feature_conf,
                 stat_file,
                 seq_len,
                 min_seq_len,
                 parall=16,
                 train_id_label="unit_id",
                 test_id_label="test_unit_id",
                 leaf_id_file="leaf.id"):
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.tree_pb_file = tree_pb_file
        self.train_sample_file = train_sample_file
        self.test_sample_file = test_sample_file
        self.train_sample_seg_cnt = train_sample_seg_cnt
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.parall = parall
        self.stat_file = stat_file
        self.feature_conf = dict()
        self.train_id_label = train_id_label
        self.test_id_label = test_id_label
        self.leaf_id_file = leaf_id_file
        with open(feature_conf) as f:
            fc = json.load(f)
            for feature_name in fc['features']:
                start = int(fc['features'][feature_name]['start'])
                end = int(fc['features'][feature_name]['end'])
                value = float(fc['features'][feature_name]['value'])
                self.feature_conf[feature_name] = (start, end, value)

    def generate(self, kv_file=None):
        self.dump_parameters()

        behavior_dict, train_sample, test_sample = self._read()
        print(repr(behavior_dict))
        stat = self._gen_train_sample(train_sample)
        # write probality stat file
        with open(self.stat_file, "wb") as f:
            for item_id in stat:
                f.write("{}, {}\n".format(item_id, stat[item_id]))
        self._gen_test_sample(test_sample)
        self._init_tree(train_sample, test_sample, stat, kv_file)

    def dump_parameters(self):
        print("\n------------------- Parameters -------------------------")
        print("train_data_file: {}".format(self.train_data_file))
        print("test_data_file: {}".format(self.test_data_file))
        print("tree_pb_file: {}".format(self.tree_pb_file))
        print("train_sample_file: {}".format(self.train_sample_file))
        print("test_sample_file: {}".format(self.test_sample_file))
        print("seq_len: {}".format(self.seq_len))
        print("min_seq_len: {}".format(self.min_seq_len))
        print("parall: {}".format(self.parall))
        print("stat_file: {}".format(self.stat_file))
        print("feature_conf: {}".format(self.feature_conf))
        print("train_id_label: {}".format(self.train_id_label))
        print("test_id_label: {}".format(self.test_id_label))
        print("--------------------------------------------------------\n")

    def _read(self):
        behavior_dict = dict()
        train_sample = dict()
        test_sample = dict()
        user_id = list()
        item_id = list()
        cat_id = list()
        behav_id = list()
        timestamp = list()

        start = time.time()
        itobj = zip([self.train_data_file, self.test_data_file],
                    [train_sample, test_sample])
        for filename, sample in itobj:
            with open(filename, 'rb') as f:
                for line in f:
                    arr = line.strip().split(',')
                    if len(arr) != 5:
                        break
                    user_id.append(int(arr[0]))
                    item_id.append(int(arr[1]))
                    cat_id.append(int(arr[2]))
                    if arr[3] not in behavior_dict:
                        i = len(behavior_dict)
                        behavior_dict[arr[3]] = i
                    behav_id.append(behavior_dict[arr[3]])
                    timestamp.append(int(arr[4]))
                sample["USERID"] = np.array(user_id)
                sample["ITEMID"] = np.array(item_id)
                sample["CATID"] = np.array(cat_id)
                sample["BEHAV"] = np.array(behav_id)
                sample["TS"] = np.array(timestamp)

                user_id = []
                item_id = []
                cat_id = []
                behav_id = []
                timestamp = []

        print("Read data done, {} train records, {} test records"
              ", elapsed: {}".format(len(train_sample["USERID"]),
                                     len(test_sample["USERID"]),
                                     time.time() - start))
        return behavior_dict, train_sample, test_sample

    def _partial_gen_train_sample(self, users,
                                  user_his_behav, filename, pipe):
        seq_len = self.seq_len
        min_len = self.min_seq_len
        stat = dict()
        with open(filename, 'wb') as f:
            for user in users:
                value = user_his_behav[user]
                count = len(value)
                if count < min_len:
                    continue
                arr = [0 for i in range(seq_len - min_len)] + \
                      [v[0] for v in value]
                for i in range(len(arr) - seq_len + 1):
                    sample = arr[i: i + seq_len]
                    f.write('{}_{}'.format(user, i))  # sample id
                    f.write('|{}_{}|'.format(user, i))  # group id
                    f.write('{}@{}:1.0'.format(
                        self.train_id_label, sample[-1]))  # label feature
                    # kvs
                    for _, feature_name in enumerate(self.feature_conf):
                        start, end, value = self.feature_conf[feature_name]
                        value = [sample[j]
                                 for j in range(start, end) if sample[j] != 0]
                        if value:
                            f.write(";")
                            f.write("{}@{}".format(feature_name,
                                                   ",".join([str(v) for v in value])))
                    f.write("||")  # dense
                    f.write("{}|".format(1.0))  # label, no ts
                    f.write('\n')
                    if sample[-1] not in stat:
                        stat[sample[-1]] = 0
                    stat[sample[-1]] += 1
        pipe.send(stat)

    def _gen_train_sample(self, train_sample):
        user_his_behav = self._gen_user_his_behave(train_sample)
        print("user_his_behav len: {}".format(len(user_his_behav)))

        users = user_his_behav.keys()
        process = []
        pipes = []
        parall = self.parall
        job_size = int(len(user_his_behav) / parall)
        if len(user_his_behav) % parall != 0:
            parall += 1
        for i in range(parall):
            a, b = mp.Pipe()
            pipes.append(a)
            p = mp.Process(
                target=self._partial_gen_train_sample,
                args=(users[i * job_size: (i + 1) * job_size],
                      user_his_behav,
                      '{}.part_{}'.format(self.train_sample_file, i),
                      b)
            )
            process.append(p)
            p.start()

        stat = dict()
        for pipe in pipes:
            st = pipe.recv()
            for k, v in st.items():
                if k not in stat:
                    stat[k] = 0
                stat[k] += v

        for p in process:
            p.join()

        # Merge partial files
        with open(self.train_sample_file, 'wb') as f:
            for i in range(parall):
                filename = '{}.part_{}'.format(self.train_sample_file, i)
                with open(filename, 'rb') as f1:
                    f.write(f1.read())

                os.remove(filename)

        # Split train sample to segments
        self._split_train_sample()
        return stat

    def _split_train_sample(self):
        segment_filenames = []
        segment_files = []
        for i in range(self.train_sample_seg_cnt):
            filename = "{}_{}".format(self.train_sample_file, i)
            segment_filenames.append(filename)
            segment_files.append(open(filename, 'wb'))

        with open(self.train_sample_file, 'rb') as fi:
            for line in fi:
                i = random.randint(0, self.train_sample_seg_cnt - 1)
                segment_files[i].write(line)

        for f in segment_files:
            f.close()

        os.remove(self.train_sample_file)

        # Shuffle
        for fn in segment_filenames:
            lines = []
            with open(fn, 'rb') as f:
                for line in f:
                    lines.append(line)
            random.shuffle(lines)
            with open(fn, 'wb') as f:
                for line in lines:
                    f.write(line)

    def _gen_user_his_behave(self, train_sample):
        user_his_behav = dict()
        iterobj = zip(train_sample["USERID"],
                      train_sample["ITEMID"], train_sample["TS"])
        for user_id, item_id, ts in iterobj:
            if user_id not in user_his_behav:
                user_his_behav[user_id] = list()
            user_his_behav[user_id].append((item_id, ts))

        for _, value in user_his_behav.items():
            value.sort(key=lambda x: x[1])

        return user_his_behav

    def _gen_test_sample(self, test_sample):
        user_his_behav = self._gen_user_his_behave(test_sample)
        with open(self.test_sample_file, 'wb') as f:
            for user, value in user_his_behav.items():
                if len(value) / 2 + 1 < self.min_seq_len:
                    continue

                mid = int(len(value) / 2)
                left = value[:mid][-self.seq_len + 1:]
                right = value[mid:]
                left = [0 for i in range(self.seq_len - len(left) - 1)] + \
                       [l[0] for l in left]

                f.write('{}_{}'.format(user, 'T'))  # sample id
                f.write('|{}_{}|'.format(user, 'T'))  # group id
                labels = ','.join(['{}:1.0'.format(item[0]) for item in right])
                f.write('{}@{}'.format(self.test_id_label, labels))
                # kvs
                for _, feature_name in enumerate(self.feature_conf):
                    start, end, value = self.feature_conf[feature_name]
                    value = [left[j]
                             for j in range(start, end) if left[j] != 0]
                    if value:
                        f.write(";")
                        f.write("{}@{}".format(feature_name,
                                               ",".join([str(v) for v in value])))
                f.write("||")  # dense
                f.write("1.0|")
                f.write('\n')

    def _init_tree(self, train_sample, test_sample, stat, kv_file):
        class Item:
            def __init__(self, item_id, cat_id):
                self.item_id = item_id
                self.cat_id = cat_id
                self.code = 0

            def __lt__(self, other):
                return self.cat_id < other.cat_id or \
                    (self.cat_id == other.cat_id and
                     self.item_id < other.item_id)

        items = []
        item_id_set = set()
        for sample in [train_sample, test_sample]:
            iterobj = zip(sample["ITEMID"], sample["CATID"])
            for item_id, cat_id in iterobj:
                if item_id not in item_id_set:
                    item_id_set.add(item_id)
                    items.append(Item(item_id, cat_id))

        with open(self.leaf_id_file, 'w') as f:
          for item_id in item_id_set:
            f.write('{}\n'.format(item_id))

        del item_id_set
        items.sort()

        def gen_code(start, end, code):
            if end <= start:
                return
            if end == start + 1:
                items[start].code = code
                return
            mid = int((start + end) / 2)
            gen_code(mid, end, 2 * code + 1)
            gen_code(start, mid, 2 * code + 2)

        gen_code(0, len(items), 0)
        ids = np.array([item.item_id for item in items])
        codes = np.array([item.code for item in items])
        data = np.array([[] for i in range(len(ids))])
        builder = tree_builder.TreeBuilder(self.tree_pb_file)
        builder.build(ids, codes, data, stat=stat, kv_file=kv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distribute tree initialization")
    parser.add_argument("--train_data_file", required=True,
                        help="filename of train data")
    parser.add_argument("--test_data_file", required=True,
                        help="filename of test data")
    parser.add_argument("--tree_pb_file", required=True,
                        help="output filename of the tree pb")
    parser.add_argument("--train_sample_file", required=True,
                        help="output filename of train sample")
    parser.add_argument("--test_sample_file", required=True,
                        help="output filename of test sample")
    parser.add_argument("--train_sample_seg_cnt", default=20,
                        help="count of train sample segments file")
    parser.add_argument("--feature_conf",
                        required=True, help="feature config file, json format")
    parser.add_argument("--stat_file",
                        required=True, help="filename of probality stat file")
    parser.add_argument("--seq_len", type=int,
                        help="sequence length of the sample record",
                        default=70)
    parser.add_argument("--min_seq_len", type=int,
                        help="Min length of the sample sequence record",
                        default=8)
    parser.add_argument("--kv_file",
                        help="output filename of the tree by key value format",
                        default=None)
    parser.add_argument('--parall',
                        type=int, help="parall process used", default=16)
    parser.add_argument('--train_id_label',
                        help="train id feature name", default='train_unit_id')
    parser.add_argument('--test_id_label',
                        help="test id feature name", default='test_unit_id')
    parser.add_argument('--leaf_id_file',
                        help="filename of leaf item id", default='leaf.item')
    argments = parser.parse_args()
    generator = Generator(argments.train_data_file,
                          argments.test_data_file,
                          argments.tree_pb_file,
                          argments.train_sample_file,
                          argments.test_sample_file,
                          argments.train_sample_seg_cnt,
                          argments.feature_conf,
                          argments.stat_file,
                          argments.seq_len,
                          argments.min_seq_len,
                          argments.parall,
                          argments.train_id_label,
                          argments.test_id_label,
                          argments.leaf_id_file)
    generator.generate(argments.kv_file)
