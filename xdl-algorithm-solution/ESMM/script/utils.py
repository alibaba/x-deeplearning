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

from __future__ import print_function
import numpy as np

class Auc(object):
    def __init__(self, name=''):
        self.name = name
        self.valid_batch_count = 0
        self.auc_value = 0.

    def add(self, preds, labels):
        raw_arr = np.column_stack((preds, labels))
        raw_arr = sorted(raw_arr, key=lambda d: d[0])

        auc = 0.
        fp1, tp1, fp2, tp2 = 0., 0., 0., 0.
        for record in raw_arr:
            fp2 += 1 - record[1]
            tp2 += record[1]
            auc += (fp2 - fp1) * (tp2 + tp1)
            fp1, tp1 = fp2, tp2

        threshold = len(preds) - 1e-3
        if tp2 > threshold or fp2 > threshold:
            print('%s invalid batch' % self.name)
            return

        if tp2 * fp2 > 0.0:
            res = (1.0 - auc / (2.0 * tp2 * fp2))
            self.auc_value +=  res
            self.valid_batch_count += 1
            print('%s_batch_auc=' % self.name, res)

    def add_with_filter(self, preds, labels, pos):
        self.add(preds[pos], labels[pos])

    def show(self):
        if self.valid_batch_count == 0:
            print('no valid batch, cannot calculate auc for', self.name)
        else:
            print('total auc of %s: ' % self.name,
                    self.auc_value / self.valid_batch_count)
