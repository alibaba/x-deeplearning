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

import os
import re

class DataSharding(object):
    def __init__(self, fs):
        self._fs = fs
        self._full_paths = list()
        self._full_dirs = list()
        self._paths = list()

    #TODO support wildcard
    def _add_path(self, path):
        if self._fs.is_dir(path):
            lst = self._fs.dir(path)
            assert len(lst) > 0
            self._full_dirs.append([path, lst])
            print "data parallel for dir: ", path
        elif self._fs.is_reg(path):
            self._full_paths.append(path)
            print "data parallel for file: ", path
        else:
            print "data parallel for re: ", path
            fname = os.path.basename(path)
            assert len(fname) > 0
            dname = os.path.dirname(path)
            assert len(dname) > 0

            lst = self._fs.dir(dname)
            assert len(lst) > 0, "error, empty dir %s"%dname
            #print "re lst", lst

            pattern = re.compile(path)
            relst = list()
            for f in lst:
                m = pattern.match(f)
                #print "re ", f, " m=", m
                if m is not None:
                    relst.append(f)

            self._full_dirs.append([path, relst])


    def add_path(self, path):
        if isinstance(path, basestring):
            self._add_path(path)
            return
        for p in path:
            self._add_path(p)

    def list(self):
        out = list()
        out.extend(self._full_paths)
        for dname, paths in self._full_dirs:
            out.extend(paths)
        return out

    def partition(self, rank, size):
        assert size > 0 and rank >= 0 and rank < size
        assert len(self._full_paths) == 0 or len(self._full_paths) >= size, "please reduce task_num to fit paths size=%d" % len(self._full_paths)

        if len(self._paths):
            return self._paths

        for i in range(len(self._full_paths)):
            if i % size != rank:
                #print "skip ",self._full_paths[i]
                continue
            self._paths.append(self._full_paths[i])

        for dname, paths in self._full_dirs:
            assert len(paths) == 0 or len(paths) >= size, "please reduce task_num to fit paths size=%d dir=%s" % (len(paths), dname)
            for i in range(len(paths)):
                if i % size != rank:
                    #print dname," skip ",paths[i]
                    continue
                self._paths.append(paths[i])

        return self._paths

class SwiftSharding(object):
    def __init__(self, client_config):
        self._client_config = client_config

    def partition(self, rank, size, threads, max_range=65536):
        assert size > 0 and rank >= 0 and rank < size and threads > 0
        worker_step = max_range / size
        res = []
        start = rank * worker_step
        local_step = worker_step if rank < size - 1 else max_range - start

        thread_step = local_step / threads
        for i in xrange(threads):
            thread_start = start + i * thread_step
            thread_local_step = thread_step if i < threads - 1 else local_step - i * thread_step
            thread_client_config = self._client_config + \
                     ";from={};to={}".format(thread_start, thread_start + \
                     thread_local_step - 1)
            res.append(thread_client_config)

        return res

