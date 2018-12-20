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

import java.io.IOException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.Date;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.conf.YarnConfiguration;

import com.alibaba.xdl.meta.Meta.ContainerMeta;
import com.alibaba.xdl.meta.Meta.Status;

public class MetaUtils {

  public static String getHostName() {
    if (System.getenv("COMPUTERNAME") != null) {
      return System.getenv("COMPUTERNAME");
    } else {
      return getHostNameForLiunx();
    }
  }

  public static String getHostNameForLiunx() {
    try {
      return (InetAddress.getLocalHost()).getHostName();
    } catch (UnknownHostException uhe) {
      String host = uhe.getMessage();
      if (host != null) {
        int colon = host.indexOf(':');
        if (colon > 0) {
          return host.substring(0, colon);
        }
      }
      return "UnknownHost";
    }
  }

  public static String getMetaPath(Configuration conf, String applicationId) throws IOException {
    String userName = UserGroupInformation.getCurrentUser().getShortUserName();
    return new StringBuilder(conf.get(YarnConfiguration.NM_REMOTE_APP_LOG_DIR)).append("/").append(userName).append("/")
        .append(conf.get(YarnConfiguration.NM_REMOTE_APP_LOG_DIR_SUFFIX)).append("/").append(applicationId).toString();
  }

  public static Meta newMeta(int serverNum, int workerNum, int schedulerNum) {
    ContainerId containerId = getContainerId();

    Meta meta = new Meta();
    meta.status = Status.RUNNING;
    meta.applicationId = containerId.getApplicationAttemptId().getApplicationId().toString();
    meta.serverNum = serverNum;
    meta.workerNum = workerNum;
    meta.schedulerNum = schedulerNum;

    ContainerMeta cm = new ContainerMeta();
    cm.containerId = containerId.toString();
    cm.hostName = MetaUtils.getHostName();
    cm.status = Status.RUNNING;
    cm.role = "AppMaster";
    cm.startTime = new Date();
    cm.stderr = System.getenv(ApplicationConstants.Environment.LOG_DIRS.toString()) + "/stderr";
    cm.stdout = System.getenv(ApplicationConstants.Environment.LOG_DIRS.toString()) + "/stdout";

    meta.containers.put(cm.containerId, cm);

    return meta;
  }

  public static ContainerId getContainerId() {
    Map<String, String> envs = System.getenv();
    String containerIdString = envs.get(ApplicationConstants.Environment.CONTAINER_ID.toString());
    if (containerIdString == null) {
      throw new IllegalArgumentException("ContainerId not set in the environment");
    }

    return ContainerId.fromString(containerIdString);
  }

  public static String getApplicationId() {
    return getContainerId().getApplicationAttemptId().getApplicationId().toString();
  }
}
