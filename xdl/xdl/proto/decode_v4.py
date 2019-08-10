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
import sample_v4_pb2 as io


def on_meta(path):
    with open(path) as fob:
        contents = fob.read()

        print len(contents)

        sm = io.SampleMeta()
        sm.ParseFromString(contents)

        print sm

def on_data(path, count):
    with open(path) as fob:
        sg = io.SampleGroup()
        for i in range(count):
            bsize = fob.read(4)
            size, = struct.unpack('I', bsize)
            assert size != 0 and size < 256000

            contents = fob.read(size)
            sg.ParseFromString(contents)

            print sg
            print "\n"


if len(sys.argv) < 2:
    print "Usage: %s <path> [count=1]"
    sys.exit(1)

path = sys.argv[1]
count = 1
if len(sys.argv) == 3:
    count = int(sys.argv[2])


if 'meta' in path:
    on_meta(path)
else:
    on_data(path, count)

