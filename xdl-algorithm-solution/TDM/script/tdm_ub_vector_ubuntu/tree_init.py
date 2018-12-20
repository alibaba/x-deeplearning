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

# !/usr/bin/env python

import xdl

import os
import argparse
import json
import subprocess

from cluster import Generator


def hdfs_download(url, dst):
  '''Download HDFS file specified by the url to the dst'''

  try:
    os.remove(dst)
  except:
    pass

  command = "hdfs dfs -get {url} {dst}".format(url=url, dst=dst)
  print(command)
  try:
    retcode = subprocess.call(command, shell=True)
  except:
    retcode = -1

  if retcode != 0:
    raise RuntimeError("Download hdfs file from {url} failed".format(url=url))


def hdfs_upload(localfile, url, over_write=False):
  '''Put local file or directory to the HDFS directory specified by the url'''

  options = ""
  if over_write:
    options += "-f"
  command = "hdfs dfs -put {options} {localfile} {url}".format(
      options=options, localfile=localfile, url=url)
  print(command)
  try:
    retcode = subprocess.call(command, shell=True)
  except:
    retcode = -1

  if retcode != 0:
    raise RuntimeError("Upload {file} to {url} failed".format(
        file=localfile, url=url))


def train(config):
  '''Tain Loop for TDM Algorithm'''

  train_rawdata_url = config["train_rawdata_url"]
  test_rawdata_url = config["test_rawdata_url"]
  data_dir = os.path.join(DIR, config['data_dir'])
  raw_train_data = os.path.join(data_dir, train_rawdata_url.split('/')[-1])
  raw_test_data = os.path.join(data_dir, test_rawdata_url.split('/')[-1])
  tree_filename = os.path.join(data_dir, config['tree_filename'])
  train_sample = os.path.join(data_dir, config['train_sample'])
  test_sample = os.path.join(data_dir, config['test_sample'])
  stat_file = os.path.join(data_dir, config['stat_file'])
  feature_conf = os.path.join(data_dir, config['feature_conf'])
  item_id_file =  os.path.join(data_dir, config['item_id_file'])

  print("Start to generating initialization data")
  # Download the raw data
  hdfs_download(train_rawdata_url, raw_train_data)
  hdfs_download(test_rawdata_url, raw_test_data)

  generator = Generator(raw_train_data,
                        raw_test_data,
                        tree_filename,
                        train_sample,
                        test_sample,
                        config["train_sample_segments"],
                        feature_conf,
                        stat_file,
                        config['seq_len'],
                        config['min_seq_len'],
                        config['parall'],
                        config['train_id_label'],
                        config['test_id_label'],
                        item_id_file)
  generator.generate()

  # Upload generating data to hdfs
  hdfs_upload(data_dir, config["upload_url"], True)


DIR = os.path.split(os.path.realpath(__file__))[0]
config_file = os.path.join(DIR, 'data/tdm.json')

if xdl.get_task_index() == 0:
  train(json.load(open(config_file)))
