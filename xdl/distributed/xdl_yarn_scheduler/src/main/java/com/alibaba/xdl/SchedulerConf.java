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

import java.util.Map;

public class SchedulerConf {

  public static class RoleResource {
    public int instance_num = 4;
    public int cpu_cores = 6;
    public long gpu_cores = 0;
    public long memory_m = 4096;
  }

  public static class ExtendRoleResource {
    public int instance_num = 4;
    public int cpu_cores = 6;
    public long gpu_cores = 0;
    public long memory_m = 4096;
    public String script = null;
    public boolean xdl_worker = false;
  }

  public static class AutoRebalance {
    public boolean enable = false;
    public String meta_dir = null;
  }
  
  public static class Docker {
    public String registry;
    public String user;
    public String password;
    public String image;
  }

  public String job_name;
  public Docker docker;
  public String script;
  public String dependent_dirs;

  public RoleResource worker;
  public RoleResource ps;
  public Map<String, ExtendRoleResource> extend_role = null;

  public boolean bind_cores = false;
  public AutoRebalance auto_rebalance;

  public String scheduler_queue = "default";
  public int min_finish_worker_num;
  public float min_finish_worker_rate = 90;
  public int max_failover_times = 20;
  public int max_local_failover_times = 3;
  public int max_failover_wait_secs = 30 * 60;

  public String env_params;
  public String volumns;
}
