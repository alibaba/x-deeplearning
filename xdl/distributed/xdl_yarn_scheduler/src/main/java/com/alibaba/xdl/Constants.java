/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.alibaba.xdl;

public class Constants {

  public static String LOG_VIEW_HOST_CONFIG_NAME = "logview.host";

  public static String OPTION_VALUE_SEPARATOR = ",";

  public final static long APPLICATION_MASTER_MEMORY = 256;

  public final static int APPLICATION_MASTER_CORE = 1;

  public final static String XDL_YARN_APP_TYPE = "XDL";

  public final static String ZOO_KEEPER_PATH_ROOT = "/xdl";

  public final static String PS_JOB_NAME = "ps";

  public final static String WORKER_JOB_NAME = "worker";

  public final static String SCHEDULER_JOB_NAME = "scheduler";

  public final static String XDL_WORK_DIR_IN_DOCKER = "/home/xdl";

  public final static String XDL_WORK_DIR_PATTERN_IN_DOCKER = "/home/%s/xdl";

  public final static String XDL_PYTHON_DIR_IN_DOCKER = "/home/xdl/python";

  public final static String XDL_PYTHON_DIR_PATTERN_IN_DOCKER = "/home/%s/xdl/python";

  public final static int DOCKER_DEFAULT_PORT = 8888;

  public final static String DOCKER_NETWORK_MODE = "host";

  public final static String PYTHON_PATH_ENV_NAME = "PYTHONPATH";

  public final static String LINUX_ROOT_USER = "root";

  public final static String NULL_VALUE = "NULL";

  public final static int CPU_PERIOD_BASE = 100000;
}
