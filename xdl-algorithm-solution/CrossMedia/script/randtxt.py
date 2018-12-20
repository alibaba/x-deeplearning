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
import time
import random

kSEG = '|';
kNAM = '@';
kFEA = ';';
kVAL = ',';
kKEY = ':';

'''
skey
sgroup
sparse
dense
label
ts
'''

N = 1024
N_SPARSE = 4
N_DENSE = 4
N_LABEL = 1
SPACE_SIZE = 1024

def on_feature(tag, n, is_sparse):
    out = tag + '@'
    if is_sparse:
        n = random.randint(1, n)
    for i in range(n):
        if i != 0:
            out += ','
        if is_sparse:
            out += "%d:%f"%(random.randint(1, SPACE_SIZE), random.random())
        else:
            out += "%f"%random.random()
    return out

def on_feature_line(tag, n, m, is_sparse):
    out = "";
    for i in range(n):
        if i != 0:
            out += ';'
        out += on_feature("%s%d"%(tag,i), m, is_sparse)
    return out;

def on_label(n):
    out = "";
    for i in range(n):
        if i != 0:
            out += ','
        out += "%f"%random.random()
    return out

for i in range(N):
    print "skey%d|gkey%d|%s|%s|%s|%d"%(i, i,
                                       on_feature_line("user", N_SPARSE, 4, True)
                                       + ";" + on_feature_line("user_img", 1, 31, True),
                                       on_feature_line("ad", N_DENSE, 1, False)
                                       + ";" + on_feature_line("ad_img", 1, 1, True),
                                       on_label(N_LABEL), time.time())
