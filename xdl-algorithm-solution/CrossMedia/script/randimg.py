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

import struct
import random

ids = range(1, 4097)
record_size = 4096

def record(i):
  fmt = 'q' + 'f' * record_size
  return struct.pack(fmt, *([i] + [random.gauss(0, 1) for k in range(record_size)]))


datadir = 'imgs/data'

files = [open(datadir + '.' + str(i), 'wb') for i in range(1024)]

for i in ids:
  files[i % 1024].write(record(i))

for f in files:
  f.close()
