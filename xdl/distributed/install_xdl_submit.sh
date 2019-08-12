#!/bin/sh
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

if [ $UID -ne 0 ]; then
    echo "Superuser privileges are required to run this script."
    echo "e.g. \"sudo sh $0\""
    exit 1
fi

cd $(dirname ${BASH_SOURCE[0]})
cp ./xdl_submit/xdl_submit.py /usr/bin/xdl_submit.py
chmod 777 /usr/bin/xdl_submit.py
cp ./xdl_yarn_scheduler/bin/xdl-yarn-scheduler-1.0.0-SNAPSHOT-jar-with-dependencies.jar /usr/bin/xdl-yarn-scheduler-1.0.0-SNAPSHOT-jar-with-dependencies.jar
chmod 777 /usr/bin/xdl-yarn-scheduler-1.0.0-SNAPSHOT-jar-with-dependencies.jar
