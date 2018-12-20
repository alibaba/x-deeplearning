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

package com.alibaba.xdl.prerun;

import java.nio.file.Paths;
import java.util.Arrays;

import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.retry.RetryOneTime;
import org.apache.hadoop.conf.Configuration;
import org.apache.zookeeper.data.Stat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.alibaba.xdl.Constants;
import com.alibaba.xdl.meta.MetaUtils;

public class PreClusterInfoHandler {

  private static final Logger LOG = LoggerFactory.getLogger(PreClusterInfoHandler.class);

  private String appId;
  private String zkAddress;
  private String clusterZKRootPath;
  private CuratorFramework zkCli;

  private volatile boolean stoped;

  public PreClusterInfoHandler(Configuration conf) throws Exception {
    this.appId = MetaUtils.getContainerId().getApplicationAttemptId().getApplicationId().toString();
    this.zkAddress = conf.get("yarn.resourcemanager.zk-address");
    this.clusterZKRootPath = Paths.get(Constants.ZOO_KEEPER_PATH_ROOT, this.appId).toString();
    this.stoped = false;

    zkCli = CuratorFrameworkFactory.newClient(zkAddress, new RetryOneTime(2000));
    zkCli.start();
  }

  public void createPaths(String zkAddress, String... paths) throws Exception {
    for (String p : paths) {
      if (zkCli.checkExists().forPath(p) == null) {
        zkCli.create().creatingParentsIfNeeded().forPath(p);
      }
    }
  }

  public void createClusterPath() throws Exception {

    String workerZKRoot = Paths.get(clusterZKRootPath, Constants.WORKER_JOB_NAME).toString();
    String psZKRoot = Paths.get(clusterZKRootPath, Constants.PS_JOB_NAME).toString();
    String scZKRoot = Paths.get(clusterZKRootPath, Constants.SCHEDULER_JOB_NAME).toString();

    createPaths(zkAddress, clusterZKRootPath, workerZKRoot, psZKRoot, scZKRoot);

    LOG.info("App Master create zookeeper cluster path: [{}]",
        Arrays.asList(clusterZKRootPath, workerZKRoot, psZKRoot, scZKRoot));
  }

  public boolean checkForRestartPrepare() throws Exception {
    Stat stat = zkCli.checkExists().forPath(clusterZKRootPath + "/restart_prepare");
    if (stat != null) {
      return true;
    } else {
      return false;
    }
  }

  public String getJobPath(String name, int index) {
    return String.format("%s/%s/%d", clusterZKRootPath, name, index);
  }

  public void updateRestartState(String jobRole, int index) throws Exception {
    String path = getJobPath(jobRole, index);
    if (zkCli.checkExists().forPath(path) != null) {
      LOG.info("zookeeper node:[{}] has existed", path);
      return;
    } else {
      zkCli.create().forPath(path);
    }
  }

  public boolean checkRestartState(int workNumber, int psNumber) throws Exception {
    for (int i = 0; i < workNumber; ++i) {
      String path = getJobPath(Constants.WORKER_JOB_NAME, i);
      if (zkCli.checkExists().forPath(path) == null) {
        return false;
      }
    }

    for (int i = 0; i < psNumber; ++i) {
      String path = getJobPath(Constants.PS_JOB_NAME, i);
      if (zkCli.checkExists().forPath(path) == null) {
        return false;
      }
    }

    String path = getJobPath(Constants.SCHEDULER_JOB_NAME, 0);
    if (zkCli.checkExists().forPath(path) == null) {
      return false;
    }
    return true;
  }

  public void restartContainers() throws Exception {
    String del = clusterZKRootPath + "/restart_prepare";
    if (zkCli.checkExists().forPath(del) != null) {
      zkCli.delete().forPath(del);
    }

    String p = clusterZKRootPath + "/restart";
    if (zkCli.checkExists().forPath(p) == null) {
      zkCli.create().forPath(p);
    }

    LOG.info("write zk {} for containers restart!", p);
  }

  public void restartContainersPrepare() throws Exception {
    String del = clusterZKRootPath + "/restart";
    if (zkCli.checkExists().forPath(del) != null) {
      zkCli.delete().forPath(del);
    }

    String create = clusterZKRootPath + "/restart_prepare";
    if (zkCli.checkExists().forPath(create) == null) {
      zkCli.create().forPath(create);
    }
    LOG.info("write zk {} for prepare restart containers!", create);
  }

  public void removeClusterPath() throws Exception {
    removeZookeeperPath(clusterZKRootPath);
  }

  private void removeZookeeperPath(String... paths) throws Exception {
    for (String path : paths) {
      zkCli.delete().deletingChildrenIfNeeded().forPath(path);
    }
    LOG.info("App Master remove zookeeper cluster path: [{}]", clusterZKRootPath);
  }

  public void stop() {
    this.stoped = true;
    LOG.info("ClusterInfoHandler is stoped!");
  }

  public void close() {
    if (zkCli != null) {
      try {
        zkCli.close();
      } catch (Exception e) {
        LOG.error("close zk client error!", e);
      }
    }
  }

  public boolean checkForToRestart() throws Exception {
    while (true) {
      Stat stat = zkCli.checkExists().forPath(clusterZKRootPath + "/restart");
      if (stat != null) {
        break;
      }
      if (stoped) {
        return false;
      }
      try {
        Thread.sleep(1000);
      } catch (Exception e) {
      }
    }

    LOG.info("check container restart {}", clusterZKRootPath + "/restart");
    return true;
  }
}
