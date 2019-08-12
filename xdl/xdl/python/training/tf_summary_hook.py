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

import xdl
import numpy as np
import matplotlib
import os
from xdl.python.framework.session import Hook
from xdl.python.training.training_utils import get_global_step
from xdl.python.utils.collections import *

class TFSummaryHook(Hook):
    def __init__(self, config, is_training=True, save_interval=1):
        super(TFSummaryHook, self).__init__()
        self._keys = []
        self._values = []
        self._types = []
        self._bins = []
        if 'output_dir' not in config:
            raise RuntimeError('summary output directory not set')
        self._lstep = 0
        self._output_dir = config['output_dir']
        import tensorflow as tf
        self._writer = tf.summary.FileWriter(self._output_dir)
        self._save_interval = save_interval

    def before_run(self, v):
        return [self._values]

    def after_run(self, v):
        import tensorflow as tf
        self._lstep += 1
        if self._lstep % self._save_interval != 0:
            return
        values = v[0]
        for i in range(len(self._keys)):
            key = self._keys[i]
            stype = self._types[i]
            val = values[i]
            bins = self._bins[i]
            if stype == 'scalar':
                summary = tf.Summary(value=[tf.Summary.Value(tag=key,
                    simple_value=val)])
            elif stype == 'histogram':
                hist = tf.HistogramProto()
                hist.min = float(np.min(val))
                hist.max = float(np.max(val))
                hist.num = float(np.prod(val.shape))
                hist.sum = float(np.sum(val))
                hist.sum_squares = float(np.sum(val**2))
                counts, edges = np.histogram(val,bins=bins)
                edges = edges[1:]
                for edge in edges:
                    hist.bucket_limit.append(edge)
                for c in counts:
                    hist.bucket.append(c)
                summary = tf.Summary(value=[tf.Summary.Value(tag=key,
                    histo=hist)])
            elif stype == 'image':
                s = StringIO()
                matplotlib.image.imsave(s, val, format='png')
                img = tf.Summary.Image(encoded_image_string=s.getvalue(),
                        height=val.shape[0],
                        width=val.shape[1])
                summary = tf.Summary(val=[tf.Summary.Value(tag=key,
                    image=img)])
            self._writer.add_summary(summary, self._lstep)

    def summary(self, tag, value, stype='histogram', bins=10):
        self._keys.append(tag)
        self._values.append(value)
        if stype not in ['scalar', 'histogram', 'image']:
            raise RuntimeError('unknown summary type %s' % stype)
        self._types.append(stype)
        if stype == 'histogram':
            self._bins.append(bins)
        else:
            self._bins.append(None)
