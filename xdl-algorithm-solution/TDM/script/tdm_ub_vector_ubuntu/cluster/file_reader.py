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

#/usr/bin/env python


from __future__ import print_function

import sys

from multiprocessing import Pool

if sys.version_info[0] < 3:
  from copy_reg import pickle
else:
  from copyreg import pickle

from types import MethodType

def _pickle_method(method):
  func_name = method.im_func.__name__
  obj = method.im_self
  cls = method.im_class
  return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
  for cls in cls.mro():
    try:
      func = cls.__dict__[func_name]
    except KeyError:
      pass
    else:
      break
  return func.__get__(obj, cls)

class FileReader(object):
  def __init__(self, filename, paral, proc=None):
    self.filename = filename
    self.paral = paral
    self.proc = proc

  def read(self):
    pool = Pool(processes=self.paral)
    splits = self._split_job()
    return pool.map(self._inner_read, splits)

  def _split_job(self):
    file_len = 0
    with open(self.filename) as f:
      f.seek(0, 2)
      file_len = f.tell()

    splits = []
    start = 0
    size = int(file_len / self.paral)
    with open(self.filename) as f:
      for _ in range(self.paral - 1):
        f.seek(size, 1)
        f.readline()
        splits.append((start, f.tell()))
        start = f.tell()
    splits.append((start, file_len))
    return splits

  def _inner_read(self, split):
    start = split[0]
    end = split[1]
    lines = []
    print("Read file {} from {} to {}".format(
      self.filename, start, end))
    with open(self.filename) as f:
      f.seek(start)
      while f.tell() < end:
        lines.append(f.readline())
    if not self.proc:
      return lines

    return self.proc(lines)

pickle(MethodType, _pickle_method, _unpickle_method)

def process(lines):
  data = []
  for line in lines:
    data.append(int(line.strip()))
  return data

def main():
  if len(sys.argv) < 2:
    print("{} <record count>".format(sys.argv[0]))
    sys.exit(1)

  # generate a tempory test file
  filename = "/tmp/test_random"
  with open(filename, "w") as f:
    for i in range(int(sys.argv[1])):
      f.write("{}\n".format(i))

  reader = FileReader(filename, 10, process)
  d = []
  for data in reader.read():
    d += data
  d.sort()
  print(d[:20])
  print(d[-20:])

if __name__ == "__main__":
  main()
