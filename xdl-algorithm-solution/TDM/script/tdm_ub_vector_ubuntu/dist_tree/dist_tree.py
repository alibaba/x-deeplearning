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

# Copyright 2018 Alibaba Inc

from .dist_tree_api import *

class DistTree:
    def __init__(self):
        self._dist_tree_handle = DIST_TREE__API_new()

    def get_handle(self):
        return self._dist_tree_handle

    def set_prefix(self, prefix):
        self._validate()
        DIST_TREE__API_set_prefix(self._dist_tree_handle, prefix)

    def set_store(self, store):
        self._validate()
        DIST_TREE__API_set_store(self._dist_tree_handle, store)

    def set_branch(self, branch):
        self._validate()
        DIST_TREE__API_set_branch(self._dist_tree_handle, branch)

    def load(self):
        self._validate()
        DIST_TREE__API_load(self._dist_tree_handle)

    def _validate(self):
        if not self._dist_tree_handle:
            raise RuntimeError("Invalid tree state, it is not initialized")

