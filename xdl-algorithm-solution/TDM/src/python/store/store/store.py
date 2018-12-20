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

from .store_api import *

class Store:
    def __init__(self, config):
        self._client = STORE_API_new(config)
        if not self._client:
            raise RuntimeError("Create store failed, config: {}".format(config))

    def get_handle(self):
        return self._client

    def put(self, key, value):
        self._validate()
        return STORE_API_put(self._client, key, value)

    def load(self, filename):
        self._validate()
        return STORE_API_load(self._client, filename)

    def get(self, key):
        self._validate()
        value = new_string()
        sv = None
        ret = STORE_API_get(self._client, key, value)
        if ret:
            sv = string_value(value)
        free_string(value)
        return sv

    def mget(self, keys):
        self._validate()

        values = new_string_vector()
        ret = STORE_API_mget(self._client, keys, values)
        svalues = None
        if ret == 1:
            svalues = string_vector_value(values)
        free_string_vector(values)
        return svalues

    def mput(self, keys, values):
        self._validate()
        return STORE_API_mput(self._client, keys, values)

    def close(self):
        if self._client:
            STORE_API_close(self._client)
            self._client = None

    def _validate(self):
        if not self._client:
            raise RuntimeError("Store is not validated")

