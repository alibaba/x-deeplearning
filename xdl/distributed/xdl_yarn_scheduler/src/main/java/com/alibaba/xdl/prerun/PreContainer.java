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

import com.alibaba.xdl.ContainerBase;
import com.alibaba.xdl.Utils;

public class PreContainer extends ContainerBase {

  private volatile PreClusterInfoHandler clusterInfoHandler;

  public PreContainer(String tfUser, String containerJobName, String config, int jobIndex, String zookeeperConnStr,
      String zookeeperRoot, boolean needRemoveDockerRunner, String volumeDirInHdfs, String port,
      String cudaVisibleDevice, String cpuset) {
    super(tfUser, containerJobName, config, jobIndex, zookeeperConnStr, zookeeperRoot, needRemoveDockerRunner,
        volumeDirInHdfs, port, cudaVisibleDevice, cpuset);
    this.cmdCode = 1;
  }

  @Override
  protected boolean beforeRunCheck() throws Exception {
    clusterInfoHandler = new PreClusterInfoHandler(config);
    return true;
  }

  @Override
  protected boolean checkJobState() throws Exception {
    // check master to restart prepare
    if (clusterInfoHandler.checkForRestartPrepare()) {
      this.print_flag = false;
      Utils.restartLog("Begin to rebalance parameters");

      // stop processor
      this.stopXDL();

      clusterInfoHandler.updateRestartState(this.jobRoleName, this.jobIndex);

      // check master restart
      clusterInfoHandler.checkForToRestart();

      // restart process, change startCmd
      this.cmdCode = 2;
      String startCmd = genCmd();
      processor = Runtime.getRuntime().exec(new String[] { "bash", "-c", startCmd });
      Utils.restartLog("Rebalance parameters successfully");
      this.print_flag = true;
      printStreamLog(processor);
      stoped = false;
    }

    // sleep
    try {
      Thread.sleep(20 * 1000);
    } catch (Exception e) {
    }
    return true;
  }

  @Override
  protected void close() {
    if (this.clusterInfoHandler != null) {
      this.clusterInfoHandler.stop();
      this.clusterInfoHandler.close();
    }
  }

}
