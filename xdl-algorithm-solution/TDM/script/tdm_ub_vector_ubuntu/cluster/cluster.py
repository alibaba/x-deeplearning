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

import os
import time
import collections
import argparse

import multiprocessing as mp
import numpy as np

from sklearn.cluster import KMeans
import tree_builder


class Cluster:
    def __init__(self,
                 filename,
                 ofilename,
                 id_offset=None,
                 parall=16,
                 kv_file=None,
                 stat_file=None,
                 prev_result=None):
        self.filename = filename
        self.ofilename = ofilename
        self.ids = None
        self.data = None
        self.parall = parall
        self.queue = None
        self.timeout = 5
        self.id_offset = id_offset
        self.codes = None
        self.kv_file = kv_file
        self.stat_file = stat_file
        self.stat = None
        self.prev_result = prev_result

    def _read(self):
        t1 = time.time()
        ids = list()
        data = list()
        with open(self.filename) as f:
            for line in f:
                arr = line.split(',')
                if not arr:
                    break
                ids.append(int(arr[0]))
                vector = list()
                for i in range(1, len(arr)):
                    vector.append(float(arr[i]))
                data.append(vector)
        self.ids = np.array(ids)
        self.data = np.array(data)
        t2 = time.time()

        if self.stat_file:
            self.stat = dict()
            with open(self.stat_file, "rb") as f:
                for line in f:
                    arr = line.split(",")
                    if len(arr) != 2:
                        break
                    self.stat[int(arr[0])] = float(arr[1])

        print("Read data done, {} records read, elapsed: {}".format(
            len(ids), t2 - t1))

    def train(self):
        ''' Cluster data '''
        self._read()
        queue = mp.Queue()
        self.process_prev_result(queue)
        processes = []
        pipes = []
        for _ in range(self.parall):
            a, b = mp.Pipe()
            p = mp.Process(target=self._train, args=(b, queue))
            processes.append(p)
            pipes.append(a)
            p.start()

        self.codes = np.zeros((len(self.ids), ), dtype=np.int64)
        for pipe in pipes:
            codes = pipe.recv()
            for i in range(len(codes)):
                if codes[i] > 0:
                    self.codes[i] = codes[i]

        for p in processes:
            p.join()

        assert(queue.empty())
        builder = tree_builder.TreeBuilder(self.ofilename)
        builder.build(self.ids, self.codes,
                      data=self.data, stat=self.stat, kv_file=self.kv_file)

    def process_prev_result(self, queue):
        if not self.prev_result:
            queue.put((0, np.array(range(len(self.ids)))))
            return True

        di = dict()
        for i, node_id in enumerate(self.ids):
            di[node_id] = i

        indexes = []
        clusters = []
        with open(self.prev_result) as f:
            for line in f:
                arr = line.split(",")
                if arr < 2:
                    break
                ni = [di[int(m)] for m in arr]
                clusters.append(ni)
                indexes += ni
        assert len(set(indexes)) == len(self.ids), \
            "ids count: {}, index count: {}".format(len(self.ids),
                                                    len(set(indexes)))
        count = len(clusters)
        assert (count & (count - 1)) == 0, \
            "Prev cluster count: {}".format(count)
        for i, ni in enumerate(clusters):
          queue.put((i + count - 1, np.array(ni)))
        return True

    def _train(self, pipe, queue):
        last_size = -1
        catch_time = 0
        processed = False
        code = np.zeros((len(self.ids), ), dtype=np.int64)
        while True:
            for _ in range(5):
                try:
                    pcode, index = queue.get(timeout=self.timeout)
                except:
                    index = None
                if index is not None:
                    break

            if index is None:
                if processed and (last_size <= 1024 or catch_time >= 3):
                    print("Process {} exits".format(os.getpid()))
                    break
                else:
                    print("Got empty job, pid: {}, time: {}".format(
                        os.getpid(), catch_time))
                    catch_time += 1
                    continue

            processed = True
            catch_time = 0
            last_size = len(index)
            if last_size <= 1024:
                self._minbatch(pcode, index, code)
            else:
                tstart = time.time()
                left_index, right_index = self._cluster(index)
                if last_size > 1024:
                    print("Train iteration done, pcode:{}, "
                          "data size: {}, elapsed time: {}"
                          .format(pcode, len(index), time.time() - tstart))
                self.timeout = int(
                    0.4 * self.timeout + 0.6 * (time.time() - tstart))
                if self.timeout < 5:
                    self.timeout = 5

                if len(left_index) > 1:
                    queue.put((2 * pcode + 1, left_index))

                if len(right_index) > 1:
                    queue.put((2 * pcode + 2, right_index))
        process_count = 0
        for c in code:
          if c > 0:
            process_count += 1
        print("Process {} process {} items".format(os.getpid(), process_count))
        pipe.send(code)

    def _minbatch(self, pcode, index, code):
        dq = collections.deque()
        dq.append((pcode, index))
        batch_size = len(index)
        tstart = time.time()
        while dq:
            pcode, index = dq.popleft()

            if len(index) == 2:
                code[index[0]] = 2 * pcode + 1
                code[index[1]] = 2 * pcode + 2
                continue

            left_index, right_index = self._cluster(index)
            if len(left_index) > 1:
                dq.append((2 * pcode + 1, left_index))
            elif len(left_index) == 1:
                code[left_index] = 2 * pcode + 1

            if len(right_index) > 1:
                dq.append((2 * pcode + 2, right_index))
            elif len(right_index) == 1:
                code[right_index] = 2 * pcode + 2

        print("Minbatch, batch size: {}, elapsed: {}".format(
            batch_size, time.time() - tstart))

    def _cluster(self, index):
        data = self.data[index]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        labels = kmeans.labels_
        l_i = np.where(labels == 0)[0]
        r_i = np.where(labels == 1)[0]
        left_index = index[l_i]
        right_index = index[r_i]
        if len(right_index) - len(left_index) > 1:
            distances = kmeans.transform(data[r_i])
            left_index, right_index = self._rebalance(
                left_index, right_index, distances[:, 1])
        elif len(left_index) - len(right_index) > 1:
            distances = kmeans.transform(data[l_i])
            left_index, right_index = self._rebalance(
                right_index, left_index, distances[:, 0])

        return left_index, right_index

    def _rebalance(self, lindex, rindex, distances):
        sorted_index = rindex[np.argsort(distances)]
        idx = np.concatenate((lindex, sorted_index))
        mid = int(len(idx) / 2)
        return idx[mid:], idx[:mid]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree cluster")
    parser.add_argument("--embed_file", required=True,
                        help="filename of the embedded vector file")
    parser.add_argument("--out_file", required=True,
                        help="filename of the output tree pb file")
    parser.add_argument("--id_offset", default=None,
                        help="id offset of the generated tree internal node")
    parser.add_argument("--parall", type=int, default=16,
                        help="Parall execution process number")
    parser.add_argument("--kv_file", default=None,
                        help="filename of the output tree kv file")
    parser.add_argument("--stat_file", default=None,
                        help="filename of the probality stat file")
    parser.add_argument("--prev_result", default=None,
                        help="filename of the previous cluster reuslt")

    argments = parser.parse_args()
    t1 = time.time()
    cluster = Cluster(argments.embed_file,
                      argments.out_file,
                      argments.id_offset,
                      argments.parall,
                      argments.kv_file,
                      argments.stat_file,
                      argments.prev_result)
    cluster.train()
    t2 = time.time()
    print("Train complete successfully, elapsed: {}".format(t2 - t1))
