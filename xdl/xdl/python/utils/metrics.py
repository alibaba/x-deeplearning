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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_METRICS = {}

def add_metrics(key, value):
    global _METRICS
    _METRICS[key] = value

def get_metrics(key, value):
    global _METRICS
    if key in _METRICS:
        return _METRICS[key]
    else:
        return ''

def get_all_metrics():
    global _METRICS
    return _METRICS

def reset_metrics():
    global _METRICS
    _METRICS = {}
