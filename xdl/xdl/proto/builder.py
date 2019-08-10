# Copyright 2018 Alibaba Group. All Rights Reserved.
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
import sample_pb2


sg = sample_pb2.SampleGroup()

# feature table
ft0 = sg.feature_tables.add()
ft1 = sg.feature_tables.add()

# for each sample
for i in range(4):
    # sample id
    sg.sample_ids.append("sg%d"%i)

    # label
    ls = sg.labels.add()
    ls.label.append(1.0)
    ls.label.append(0.0)

    fl = ft0.feature_lines.add()
    fl.refer = i / 2

    # pv
    f = fl.features.add()
    f.type = sample_pb2.Feature.SPARSE
    f.name = 'pv'
    v = f.values.add()
    v.key = i
    v.value = i

    # ref -> uv
    f = fl.features.add()
    f.type = sample_pb2.Feature.REFER

    if i % 2 == 0:
        fl = ft1.feature_lines.add()
        f = fl.features.add()
        f.type = sample_pb2.Feature.SPARSE
        f.name = 'uv'
        v = f.values.add()
        v.key = i
        v.value = i / 10.


f = open(sys.argv[1], "wb")
zb = sg.SerializeToString()
l = len(zb)
print l
lb = struct.pack("I", l)

for i in range(3):
    f.write(lb)
    f.write(zb)
f.close()
