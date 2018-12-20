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
import json

if len(sys.argv) < 4:
  print('{} <train_file> <test_file> <category_file>'.format(sys.argv[0]))
  sys.exit(-1)

train_file = sys.argv[1]
test_file = sys.argv[2]
category_file = sys.argv[3]


max_cat_id = 1
cat_name_to_id = dict()
movie_cats = dict()
with open(category_file) as f:
  for line in f:
    line = line.strip()
    arr = line.split('^')
    if len(arr) != 3:
      break
    movie_id = arr[0].strip()
    # movie_title = arr[1]
    cats = [v.strip() for v in arr[2].split('|')]
    for cat in cats:
      if cat not in cat_name_to_id:
        cat_name_to_id[cat] = max_cat_id
        max_cat_id += 1
    movie_cats[movie_id] = [cat_name_to_id[v] for v in cats]

with open("testdata/category_meta.json", "wb") as f:
  f.write(json.dumps(cat_name_to_id))
  f.write("\n")
  f.write(json.dumps(movie_cats))
  f.write("\n")

for filename in [train_file, test_file]:
  with open(filename) as f:
    with open(filename + '.new', 'wb') as fo:
      for line in f:
        line = line.strip()
        arr = line.split(',')
        movie_id = arr[1]
        if movie_id not in movie_cats:
          raise RuntimeError("{} not in category".format(movie_id))
        arr.insert(2, str(movie_cats[movie_id][0]))
        fo.write(','.join(arr))
        fo.write('\n')

