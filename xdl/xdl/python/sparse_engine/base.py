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

class SparseTensor(object):
    """ a SparseTensor represent a batch of sparse samples. 
        eg: three sparse sample: [1,5,7], [2, 4, 10], [3]
            can be represented as a SparseTensor t:
            t._ids = [1,5,7,2,4,10,3]
            t._values = None
            t._segments = [3,6,7]
    """
    def __init__(self, ids, values, segments, indices=None, sidx=None, sseg=None):
        self._ids = ids
        self._values = values
        self._segments = segments
        self._indices = indices
        self._sidx = sidx
        self._sseg = sseg
        self._shape = None
        self._name = None

    def has_unique_ids(self):
        return self._indices != None

    @property
    def ids(self):
        return self._ids
    @property
    def values(self):
        return self._values
    @property
    def segments(self):
        return self._segments
    @property
    def indices(self):
        return self._indices
    @property
    def sidx(self):
        return self._sidx
    @property
    def sseg(self):
        return self._sseg

    @property
    def shape(self):
        return self._shape
    def set_shape(self, shape):
        self._shape = shape

    @property
    def name(self):
        return self._name
    def set_name(self, name):
        self._name = name


class MergedSparseTensor(object):
    """ a MergedSparseTensor represent a batch of sparse samples 
        which have multi feature groups, each feature group is a SparseTensor
        eg: three sparse sample
            sample   group1 group2
            1        [1,5]   [1]
            2        [2,4]   [3]
            3        [2]     [4,5]
            can be represented as a MergeSparseTensor t:
            t._ids = [1,5,1,2,4,3,2,4,5]
            t._values = None
            t._segments = [3,6,9]
            t._groups = [2,3,5,6,7,9]
    """
    def __init__(self, ids, values, segments, groups, indices=None, sidx=None, sseg=None):
        self._ids = ids
        self._values = values
        self._segments = segments
        self._groups = groups
        self._indices = indices
        self._sidx = sidx
        self._sseg = sseg
        self._shape = None
        self._name = None

    def has_unique_ids(self):
        return self._indices != None

    @property
    def ids(self):
        return self._ids
    @property
    def values(self):
        return self._values
    @property
    def segments(self):
        return self._segments
    @property
    def groups(self):
        return self._groups
    @property
    def indices(self):
        return self._indices
    @property
    def sidx(self):
        return self._sidx
    @property
    def sseg(self):
        return self._sseg
    @property
    def shape(self):
        return self._shape
    def set_shape(self, shape):
        self._shape = shape

    @property
    def name(self):
        return self._name
    def set_shape(self, name):
        self._name = name
