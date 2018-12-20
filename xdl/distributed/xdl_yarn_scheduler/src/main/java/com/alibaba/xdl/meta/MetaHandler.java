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
import java.io.PrintWriter;
import java.util.Date;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.codehaus.jackson.map.ObjectMapper;

import com.alibaba.xdl.meta.Meta.ContainerMeta;
import com.alibaba.xdl.meta.Meta.Status;

public class MetaHandler {

  private static final Log LOG = LogFactory.getLog(MetaHandler.class);
  private static final FsPermission permTemp = new FsPermission("777");

  private String metaDir;
  private Meta meta;
  private int serverNum;
  private int workerNum;
  private int schedulerNum;
  private Configuration conf;

  private MetaHandler() {
  }

  public static MetaHandler newInstance(int serverNum, int workerNum, int schedulerNum, Configuration conf)
      throws IOException {
    MetaHandler instance = new MetaHandler();
    instance.meta = MetaUtils.newMeta(serverNum, workerNum, schedulerNum);
    instance.serverNum = serverNum;
    instance.workerNum = workerNum;
    instance.schedulerNum = schedulerNum;
    instance.metaDir = MetaUtils.getMetaPath(conf, instance.meta.applicationId);
    instance.conf = conf;
    return instance;
  }

  public synchronized void flush() {
    PrintWriter out = null;
    FileSystem dfsCli = null;
    try {
      dfsCli = FileSystem.get(conf);
      Path dir = new Path(this.metaDir);
      if (!dfsCli.exists(dir)) {
        dfsCli.mkdirs(dir, permTemp);
      }

      String metaJson = new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(meta);
      Path metaPath = new Path(metaDir + "/meta");
      LOG.debug("meta flush to " + metaPath + ", json: " + metaJson);
      out = new PrintWriter(dfsCli.create(metaPath, true));
      out.print(metaJson);
      out.flush();
      LOG.debug("meta flush success!");
    } catch (IOException e) {
      e.printStackTrace();
      throw new RuntimeException("flush meta error!", e);
    } finally {
      try {
        out.close();
      } catch (Exception e) {
      }
      try {
        dfsCli.close();
      } catch (Exception e) {
      }
    }
  }

  public synchronized void close() {
  }

  public synchronized void updateDirs(String checkpointDir, String summaryDir) {
    this.meta.checkpointDir = checkpointDir;
    this.meta.summaryDir = summaryDir;
  }

  public synchronized void clearForJobFail(int failTimes) {
    this.meta = MetaUtils.newMeta(serverNum, workerNum, schedulerNum);
    FileSystem dfsCli = null;
    try {
      dfsCli = FileSystem.get(conf);
      Path src = new Path(metaDir + "/meta");
      Path dst = new Path(metaDir + "/meta_" + failTimes);

      dfsCli.rename(src, dst);

      LOG.info("meta rename from " + src + " to " + dst);
    } catch (IOException e) {
      e.printStackTrace();
      throw new RuntimeException("clear meta error!", e);
    }
  }

  public synchronized void addContainer(ContainerMeta cm) {
    meta.containers.put(cm.containerId, cm);
  }

  public synchronized void updateContainer(String containerId, Status status) {
    ContainerMeta cm = meta.containers.get(containerId);
    if (cm != null) {
      cm.status = status;
      cm.endTime = new Date();
    }
  }

  public synchronized void updateContainer(String containerId, String port) {
    ContainerMeta cm = meta.containers.get(containerId);
    if (cm != null) {
      cm.port = port;
    }
  }

  public synchronized void udpateJobStatus(Status status) {
    this.meta.status = status;
  }

  public synchronized void updateAppMaster(Status status) {
    this.udpateJobStatus(status);
    this.updateContainer(MetaUtils.getContainerId().toString(), status);
    this.flush();
  }

  public synchronized Meta getMeta() {
    return meta;
  }
}
