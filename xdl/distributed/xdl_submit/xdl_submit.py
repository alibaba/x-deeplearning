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

#!/home/tops/bin/python
#!/bin/env python
#encoding=utf-8

import argparse
import json
import logging
import sys
import os
import uuid
import subprocess
from subprocess import Popen

TF_JAR = "/usr/bin/xdl-yarn-scheduler-1.0.0-SNAPSHOT-jar-with-dependencies.jar"
TF_MAIN_CLASS = "com.alibaba.xdl.Client"

def main():
    args, dargs = parse_args()
    cmd = gen_client_cmd(args, dargs)
    print("CMD: %s" % cmd)
    rc = realtime_shell(cmd)
    return rc

def gen_client_cmd(args, dargs):
  u = str(uuid.uuid4())
  config = replace_config(args.config, u, dargs)
  client_cmd = "%s jar %s %s" % (get_hadoop_bin(), TF_JAR, TF_MAIN_CLASS)
  client_cmd += " -c=%s" % config
  client_cmd += " -f=%s" % TF_JAR
  client_cmd += " -uuid=%s" % u
  return client_cmd.encode('utf-8')

def get_hadoop_bin():
  env_dist = os.environ
  hadoop_home =  env_dist.get('HADOOP_HOME')
  if hadoop_home:
    bin = hadoop_home + "/bin/hadoop"
    if os.path.exists(bin):
      print("hadoop bin %s" % bin)
      return bin
  else:
    default_bin = "/usr/local/hadoop/bin/hadoop"
    if os.path.exists(default_bin):
      print("hadoop bin default %s" % default_bin)
      return default_bin
  
  print("Need HADOOP_HOME, can submit job by 'HADOOP_HOME=/xxxx/hadoop xdl_submit.py --config xxx.json")
  raise RuntimeError("run cmd error!")

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--config',
    type=str,
    default="",
    help='job json config'
  )
  FLAGS, unparsed = parser.parse_known_args()
  DARGS = parse_dargs(unparsed)
  if len(FLAGS.config) == 0:
    raise RuntimeError("config is None")
  return FLAGS, DARGS

def parse_dargs(args):
  result = {}
  if not args:
    return result
  for arg in args:
    if "=" not in arg: 
      continue 
    i = arg.index("=")
    if arg.startswith('-D') and i != -1:
      name = arg[2:i]
      value = arg[i+1:]
      result[name] = value
  return result

def replace_config(config, uuid, dargs):
  if dargs:
    file = open(config)
    content = ''
    for line in  file.readlines(): 
      content += line +'\n'
    for key in dargs:
      content = content.replace("${"+key+"}", dargs[key]);
      
    # create new file
    local_dir = '/tmp/xdl_local/' + uuid
    os.makedirs(local_dir)
    new_config = local_dir + "/" + os.path.basename(config)  
    output = open(new_config, "w")
    output.write(content)
    output.close()
    return new_config
  else:
    return config

def realtime_shell(cmd):
  logger = logging.getLogger(__name__)
  logger.debug("begin to run cmd:%s", str(cmd))

  rc = 1
  try:
    p = Popen(cmd, shell = True, stdout = sys.stdout, stderr = sys.stderr)
    p.communicate()
    rc = p.returncode
    logger.debug("run cmd %s, rc:%s.", "success" if rc==0 else "fail", rc)
    return rc
  except Exception, msg:
    logger.error("run cmd faild, rc:%s, err_msg:%s", rc, str(msg))
    return 1

# main
if __name__ == "__main__":
  exit_code = 0
  try:
    exit_code = main()
  except Exception, e:
    import traceback
    traceback.print_exc()
    exit_code = 2
  sys.exit(exit_code)

