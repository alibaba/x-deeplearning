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

import com.alibaba.xdl.AppMasterBase;
import com.alibaba.xdl.Utils;
import com.alibaba.xdl.meta.MetaUtils;

public class PreAppMaster extends AppMasterBase {

  public PreAppMaster(String basePath, String user, String config, String volumes) {
    super(basePath, user, config, volumes);
  }

  private PreClusterInfoHandler clusterInfoHandler;
  private boolean preRunFlag = true;

  private void delCheckpointDir() throws Exception {
    String ckDir = this.jobConf.getCheckpointOutputDir();
    try {
      Utils.removeHdfsDirs(this.yarnConf, ckDir);
    } catch (Exception e) {
    }
  }

  @Override
  protected void init() throws Exception {
    this.clusterInfoHandler = new PreClusterInfoHandler(this.yarnConf);
    this.clusterInfoHandler.createClusterPath();
  }

  @Override
  protected void preRun() throws Exception {
    boolean fileFlag = false;
    if (this.schedulerConf.auto_rebalance.meta_dir == null) {
      String defaultFs = this.yarnConf.get("fs.defaultFS");
      String metaDir = defaultFs + MetaUtils.getMetaPath(this.yarnConf, this.appId) + "/meta_info";
      fileFlag = Utils.existsHdfsFile(yarnConf, metaDir);
    } else {
      fileFlag = Utils.existsHdfsFile(yarnConf, this.schedulerConf.auto_rebalance.meta_dir);
    }
    if (fileFlag == true && preRunFlag == true) {
      // sleep for waiting meta_info file finish
      try {
        Thread.sleep(30 * 1000);
      } catch (Exception e) {
      }

      LOG.info("Finish collecting meta info, begin to restart to normal run.");

      this.clusterInfoHandler.restartContainersPrepare();

      // sleep for waiting container stop docker
      try {
        Thread.sleep(60 * 1000);
      } catch (Exception e) {
      }

      while (!this.clusterInfoHandler.checkRestartState(this.workNumber, this.psNumber)) {
        try {
          Thread.sleep(5 * 1000);
        } catch (Exception e) {
        }
      }
      
      this.delCheckpointDir();
      this.clusterInfoHandler.restartContainers();
      preRunFlag = false;
    }
  }

  @Override
  protected void close() throws Exception {
    removeZkRootPath();
    closeZkCli(); // close zkcli of app master
    this.clusterInfoHandler.removeClusterPath();
    this.clusterInfoHandler.stop();
    this.clusterInfoHandler.close();
  }
}
