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

#  Copyright 2018 Alibaba Inc. All Rights Conserved

import argparse
import random


class DataCutter(object):
    def __init__(self, inp, train, test, number):
        self._input = inp
        self._train = train
        self._test = test
        self._number = number

    def cut(self):
        user_behav = dict()
        user_ids = list()
        with open(self._input) as f:
            for line in f:
                arr = line.strip().split(',')
                if len(arr) != 5:
                    break

                if arr[0] not in user_behav:
                    user_ids.append(arr[0])
                    user_behav[arr[0]] = list()

                user_behav[arr[0]].append(line)

        random.shuffle(user_ids)
        test_user_ids = user_ids[:self._number]
        train_user_ids = user_ids[self._number:]

        # write train data set
        with open(self._train, 'wb') as f:
            for uid in train_user_ids:
                for line in user_behav[uid]:
                    f.write(line)

        with open(self._test, 'wb') as f:
            for uid in test_user_ids:
                for line in user_behav[uid]:
                    f.write(line)


if __name__ == '__main__':
    _PARSER = argparse.ArgumentParser(
        description="DataCutter: cut data to train set and test set")
    _PARSER.add_argument("--input", required=True, help="input filename")
    _PARSER.add_argument("--train", required=True,
                         help="filename of output train set")
    _PARSER.add_argument("--test", required=True,
                         help="filename of output test set")
    _PARSER.add_argument("--number", required=True, type=int,
                         help="number of users for test set")
    _ARGS = _PARSER.parse_args()
    DataCutter(_ARGS.input, _ARGS.train, _ARGS.test, _ARGS.number).cut()


