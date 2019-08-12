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

from xdl.python.pybind import hdfs_read, hdfs_write, get_file_system

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

class FileSystemClient(object):
  def __init__(self, fs_type, namenode, reader_path=None, writer_path=None):
    self._client = get_file_system(fs_type, namenode)
    self._reader_path = reader_path
    self._writer_path = writer_path
    self._reader = None
    self._writer = None
    if reader_path is not None:
      self._reader = self._client.get_ant(reader_path, 'r')
    if writer_path is not None:
      self._writer = self._client.get_ant(writer_path, 'w')

  def read(self, path=None):
    if path is None:
      path = self._reader_path
    if path is None:
      print('ERROR: cannot read without reader path')
      return
    # TODO

  def write(self, msg, size, path=None):
    if path is None:
      writer = self._writer
    else:
      writer = self._client.get_ant(path, 'w')
    if writer is None:
      print('ERROR: cannot write without writer path')
      return
    res = writer.write(msg, size)
    if res == -1:
      print('ERROR: write to swift failed')
