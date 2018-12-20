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

from xdl.python.pybind import hdfs_read, hdfs_write

def write_string_to_file(name, content):
    if name.startswith("hdfs://"):
        hdfs_write(name, content)
    else:
        with open(name, 'w') as f:
            f.write(content)

def read_string_from_file(name):
    if name.startswith("hdfs://"):
        return hdfs_read(name)
    else:
        with open(name, 'r') as f:
            return f.read()
