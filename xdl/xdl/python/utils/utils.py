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
import xdl
import subprocess
def decode_strings_from_buf(addrs, lens):
    """ get string at index by addrs and lens
    """
    strings = list()
    assert len(addrs) == len(lens)
    for i in range(len(lens)):
        if lens[i] > 0:
            strings.append("".join(map(chr, addrs[i, 0:lens[i]])))
    return strings

def bytes_to_int(bs):
    bs = bytearray(bs)
    return sum((int(b) << (k * 8) for k, b in enumerate(bs)))

def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    out = ''
    for o in iter(p.stdout.readline, b''):
        out += o
    return out


def read_checkpoint_list():
    checkpoint_meta_file = xdl.get_ckpt_dir() + '/checkpoints'
    print 'checkpoints file', checkpoint_meta_file
    print run_command("$HADOOP_HOME/bin/hadoop fs -get " +  checkpoint_meta_file + ' ./checkpoints ')
    chcks = []
    if not os.path.isfile('./checkpoints') :
        print './checkpoints not exists'
        return []
    size_bytes = 8
    with open('checkpoints', 'rb') as fd:
        bs = fd.read(size_bytes)
        if bs is not None:
            checkpoins_size = bytes_to_int(bs)

            i = 0
            while i < checkpoins_size:
                bs =  bytes_to_int(fd.read(size_bytes))
                p = str(fd.read(bs))
                chcks.append((bs, p))
                i += 1
    run_command('rm checkpoints')
    return chcks
