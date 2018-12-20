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

#/usr/bin/env python

from __future__ import print_function

import sys
import argparse
import time

import numpy as np

from file_reader import FileReader


class Balancer(object):
  def __init__(self, input_file, output_file):
    self._input_file = input_file
    self._output_file = output_file

  def _read(self):
    cluster_indexes = dict()
    distances = dict()
    count = 0
    start = time.time()
    reader = FileReader(self._input_file, 64, self._process)
    for ci, ds, c in reader.read():
      distances.update(ds)
      count += c
      for x in ci:
        if x not in cluster_indexes:
          cluster_indexes[x] = np.array(ci[x], dtype="int64")
        else:
          cluster_indexes[x] = np.append(cluster_indexes[x], ci[x])

    assert count == len(distances), \
      "ids count: {}, record count: {}".format(
        len(distances), count)
    print("Read data done, read {} records, elapsed: {}".format(
      count, time.time() - start))
    return cluster_indexes, distances, count

  def _process(self, lines):
    cluster_indexes = dict()
    distances = dict()
    count = 0
    for line in lines:
      arr = [float(v) for v in line.split(",")]
      if len(arr) < 2:
        break
      node_id = int(arr[0])
      index = int(arr[1])
      if index not in cluster_indexes:
        cluster_indexes[index] = list()
      cluster_indexes[index].append(node_id)
      distances[node_id] = np.array(arr[2:], dtype="float16")
      count += 1
    print("Processed {} lines".format(count))
    return cluster_indexes, distances, count

  def balance(self):
    cluster_indexes, distances, count = self._read()
    centers = 0
    for x in distances:
      centers = len(distances[x])
      break
    average = int(count / centers)
    moves = [0 for i in range(centers)]
    residual = count % centers
    print("Start balance, {} centers, average count: {}".format(
      centers, average))
    for i in range(centers):
      moves[i] -= average
      if i in cluster_indexes:
        moves[i] += len(cluster_indexes[i])
      else:
        cluster_indexes[i] = np.array([], dtype="int64")

    for index in cluster_indexes:
      dis = np.array(
        [distances[nid][index] for nid in cluster_indexes[index]])
      si = np.argsort(dis)
      cluster_indexes[index] = cluster_indexes[index][si]

    for i in range(residual):
      moves[i] -= 1
    assert(0 == sum(moves))

    for i in range(centers):
      if moves[i] > 0:
        move_outs = cluster_indexes[i][-moves[i]:]
        cluster_indexes[i] = cluster_indexes[i][:-moves[i]]
        for node_id in move_outs:
          index =  np.array(range(centers))
          sort_idx = np.argsort(distances[node_id])
          index = index[sort_idx]
          for j in index:
            if moves[j] < 0:  # moves node_id from i to j
              cluster_indexes[j] = np.append(cluster_indexes[j], node_id)
              moves[j] += 1
              break

    with open(self._output_file, 'wb') as f:
      for i in range(centers):
        for j, node_id in enumerate(cluster_indexes[i]):
          if j > 0:
            f.write(", ")
          f.write("{}".format(node_id))
        f.write("\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Tree Balancer")
  parser.add_argument("--input", required=True,
                      help="filename of the input file")
  parser.add_argument("--output", required=True,
                      help="filename of the output file")
  argments = parser.parse_args()
  balancer = Balancer(argments.input, argments.output)
  balancer.balance()

