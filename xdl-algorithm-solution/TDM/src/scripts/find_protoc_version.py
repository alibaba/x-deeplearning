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
import re

header_file = sys.argv[1]
f = open(header_file)
version = None
for line in f:
    m = re.match(r'#define\s+GOOGLE_PROTOBUF_VERSION\s+(\d+)', line)
    if m:
        version = m.group(1)
        break
f.close()

if version:
    v = int(version)
    major = int(v / 1000000)
    minor = float(v - major * 1000000) / 1000
    print("protobuf-{0}.{1:.1f}".format(major, minor))
