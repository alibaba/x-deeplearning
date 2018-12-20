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

from cluster import Cluster


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

  data_dir = os.path.join(DIR, config['data_dir'])
  tree_filename = os.path.join(data_dir, config['tree_filename'])
  stat_file = os.path.join(data_dir, config['stat_file'])

  print("Start to cluster tree")
  # Download item id
  upload_dir = os.path.join(config['upload_url'], os.path.split(data_dir)[-1])
  item_id_url = os.path.join(upload_dir, config['item_id_file'])
  item_id_file = os.path.join(data_dir, 'item.id')
  hdfs_download(item_id_url, item_id_file)
  model_embed_tmp = os.path.join(data_dir, 'model.embed.tmp')
  hdfs_download(config['model_url'] + '/item_emb', model_embed_tmp)

  # Read max item id from item id file
  max_item_id = 0
  with open(item_id_file) as f:
    for line in f:
      item_id = int(line)
      if item_id > max_item_id:
        max_item_id = item_id
  max_item_id += 1

  model_embed = os.path.join(data_dir, 'model.embed')
  item_count = 0
  id_set = set()
  with open(model_embed_tmp) as f:
    with open(model_embed, 'wb') as fo:
      for line in f:
        arr = line.split(",")
        item_id = int(arr[0])
        if (len(arr) > 2) and (item_id < max_item_id) and (item_id not in id_set):
          id_set.add(item_id)
          item_count += 1
          fo.write(line)

  os.remove(model_embed_tmp)
  print("Filer embedding done, records:{}, max_leaf_id: {}".format(
      item_count, max_item_id))

  # Tree clustering
  cluster = Cluster(model_embed, tree_filename,
                    parall=config['parall'], stat_file=stat_file)
  cluster.train()

  # Upload clustered tree to hdfs
  tree_upload_dir = os.path.join(config['upload_url'], os.path.split(data_dir)[-1])
  hdfs_upload(tree_filename, tree_upload_dir, over_write=True)


DIR = os.path.split(os.path.realpath(__file__))[0]
config_file = os.path.join(DIR, 'data/tdm.json')

if xdl.get_task_index() == 0:
  train(json.load(open(config_file)))
