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

package com.alibaba.xdl.meta;

import java.util.Collection;
import java.util.Date;
import java.util.Map;

import com.google.common.collect.Maps;

public class Meta {

  public static enum Status {
    READY, RUNNING, FINISHED, FAIL, KILLED
  }

  public Status status;
  public String applicationId;
  public int serverNum;
  public int workerNum;
  public int schedulerNum;
  public String checkpointDir;
  public String summaryDir;

  public Map<String, ContainerMeta> containers = Maps.newHashMap();

  public Status getStatus() {
    return status;
  }

  public String getApplicationId() {
    return applicationId;
  }

  public int getServerNum() {
    return serverNum;
  }

  public int getWorkerNum() {
    return workerNum;
  }

  public Collection<ContainerMeta> getContainers() {
    return containers.values();
  }

  public String getCheckpointDir() {
    return checkpointDir;
  }

  public String getSummaryDir() {
    return summaryDir;
  }

  public void changeStatus(Status from, Status to) {
    for (ContainerMeta meta : containers.values()) {
      if (meta.status == from) {
        meta.status = to;
      }
    }
  }

  public static class ContainerMeta {

    public String containerId;
    public String hostName;
    public String port;
    public int taskIdx;
    public Status status;
    public int curAttamptIdx;
    public String role;
    public Date startTime;
    public Date endTime;
    public int cores;
    public long memoryMB;
    public String command;
    public String stdout;
    public String stderr;

    public String getContainerId() {
      return containerId;
    }

    public String getHostName() {
      return hostName;
    }

    public String getPort() {
      return port;
    }

    public int getTaskIdx() {
      return taskIdx;
    }

    public Status getStatus() {
      return status;
    }

    public int getCurAttamptIdx() {
      return curAttamptIdx;
    }

    public String getRole() {
      return role;
    }

    public Date getStartTime() {
      return startTime;
    }

    public Date getEndTime() {
      return endTime;
    }

    public int getCores() {
      return cores;
    }

    public long getMemoryMB() {
      return memoryMB;
    }

    public String getCommand() {
      return command;
    }

    public String getStdout() {
      return stdout;
    }

    public String getStderr() {
      return stderr;
    }
  }
}
