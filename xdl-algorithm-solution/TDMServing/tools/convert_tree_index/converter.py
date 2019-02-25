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

import sys
import struct
import proto.offline_tree_pb2 as offline_tree
import proto.online_tree_pb2 as online_tree

if len(sys.argv) < 3:
  print("{command} <input_pb> <head_number>".format(command=sys.argv[0]))
  sys.exit(-1)

offsets = dict()
levels = dict()
code_node_map = dict()

head_num = int(sys.argv[2])

def get_level(code):
  level = 0
  while code > 0:
    code = int((code - 1) / 2)
    level += 1
  return level

# Read tree
print("Read tree nodes begin ...")

read_count = 0
with open(sys.argv[1], 'rb') as f:
  while True:
    bf = f.read(4)
    if len(bf) < 4:
      break
    n = struct.unpack('i', bf)[0]

    bf = f.read(n)
    if len(bf) < n:
      break

    item = offline_tree.KVItem()
    item.ParseFromString(bf)
    if len(item.key) == 8:  # Node Type
      code = struct.unpack('L', item.key[::-1])[0]
      offline_node = offline_tree.Node()
      offline_node.ParseFromString(item.value)

      level = get_level(code)
      if level not in levels:
        levels[level] = list()
      levels[level].append(code)

      node = online_tree.UINode()
      node.id = offline_node.id
      node.level = get_level(code)
      node.data.append(offline_node.probality)
      node.leaf_cate_id = offline_node.leaf_cate_id
      node.hashid = node.id
      code_node_map[code] = node

    read_count += 1
    if read_count % 500000 == 0:
      print("Read {count} tree nodes".format(count=read_count))

print("Read {count} tree nodes".format(count=read_count))
print("Read tree nodes end ...")

# Process node sequence
print("Process tree nodes begin ...")

seq = 0
seq_code_map = dict()
for level in range(len(levels)):
  code_list = levels[level]
  levels[level] = sorted(code_list)
  for i, code in enumerate(levels[level]):
    offsets[code] = seq + i
    seq_code_map[seq + i] = code
  seq += len(code_list)

nodes_number = len(offsets)

proc_count = 0
for code, node in code_node_map.items():
  node.seq = offsets[code]
  node.parent = node.seq
  if code > 0:
    parent = int((code - 1) / 2)
    node.parent = offsets[parent]

  left = 2 * code + 1
  if left in offsets:
    node.children.append(offsets[left])

  right = 2 * code + 2
  if right in offsets:
    node.children.append(offsets[right])

  proc_count += 1
  if proc_count % 500000 == 0:
    print("Process {count} tree nodes".format(count=proc_count))

meta = online_tree.UIMeta()
seq_list = range(0, nodes_number)
code_seq = [seq_code_map[seq] for seq in seq_list]

print("Process {count} tree nodes".format(count=proc_count))
print("Process tree nodes end ...")

print("Write tree nodes begin ...")

count_per_head = int(nodes_number / head_num)
if nodes_number % head_num != 0:
  count_per_head += 1

for i in range(head_num):
  tree = online_tree.UITree()
  head = online_tree.UIHead()

  head.tid = i
  head.offset = i * count_per_head
  codes = code_seq[head.offset: head.offset + count_per_head]
  head.count = len(codes)

  meta.heads.extend([head])
  tree.head.CopyFrom(head)
  tree.nodes.extend([code_node_map[code] for code in codes])

  with open('tree.dat.{index}'.format(index=i), 'wb') as f:
    f.write(tree.SerializeToString())

with open('meta.dat', 'wb') as f:
  f.write(meta.SerializeToString())

print("Write tree nodes end ...")

