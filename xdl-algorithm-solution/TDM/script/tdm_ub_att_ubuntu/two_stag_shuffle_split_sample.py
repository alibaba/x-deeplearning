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

# -*-coding:utf-8 -*-
import random

print "helloWorld"

split_num = 20
origin_sample_path = "movielens_train_sample_uniq_groupkey.dat"

# 开启split_num个文件输出
file_sample_num = []
foutputs = []
split_file_paths = []
for i in xrange(0, split_num):
    print "Create File ", origin_sample_path + "_%i.dat" % i
    split_file_paths.append(origin_sample_path + "_%i.dat" % i)
    foutputs.append(open(origin_sample_path + "_%i.dat" % i, "w"))
    file_sample_num.append(0)

# 首先，将所有数据随机输出到不同文件中
sp = 0
with open(origin_sample_path) as finput:
    for line in finput:
        # 随机数
        random_file_num = random.randint(0, split_num - 1)

        foutputs[random_file_num].write(line)
        file_sample_num[random_file_num] += 1

        sp += 1
        if sp % 10000 == 0:
            print "sp ", sp

# 将各个文件关闭
for i in xrange(0, split_num):
    foutputs[i].close()

# 重新读入各个文件，在文件内shuffle
for i in xrange(0, split_num):
    print "shuffling file", i, "start"
    samples = []
    with open(split_file_paths[i]) as finput:
        for line in finput:
            samples.append(line)

    random.shuffle(samples)

    fouput = open(split_file_paths[i], "w")
    for line in samples:
        fouput.write(line)

    fouput.close()
    print "shuffling file", i, "end"

print "end"
