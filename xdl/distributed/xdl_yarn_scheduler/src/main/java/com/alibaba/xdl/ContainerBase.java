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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.alibaba.xdl.meta.MetaUtils;

public class ContainerBase {

  private static Logger LOG = LoggerFactory.getLogger(ContainerBase.class);

  private String yarnAppId;
  private String yarnUser;
  private String containerWorkDir;
  private String zookeeperRoot;
  private String zookeeperConnStr;
  protected YarnConfiguration config;

  protected String jobRoleName;
  protected int jobIndex;
  private SchedulerConf schedulerConf;
  private String jobConfig;

  private String volumeDirInHdfs;
  private String cudaVisibleDevice;
  private String cpuset;

  private int exitCode = 0;
  private volatile boolean needRemoveDockerRunner = false;

  protected volatile Process processor;
  private volatile String dockerName;
  private volatile String stopCmd;
  private volatile String dockerLogin;
  private volatile String pullImage;
  private volatile String removeCmd;
  protected int cmdCode = 0;
  protected volatile boolean print_flag = true;

  public ContainerBase(String xdlUser, String containerJobName, String config, int jobIndex, String zookeeperConnStr,
      String zookeeperRoot, boolean needRemoveDockerRunner, String volumeDirInHdfs, String port,
      String cudaVisibleDevice, String cpuset) {
    this.jobConfig = config;
    this.schedulerConf = Utils.parseSchedulerConf(config);
    this.yarnAppId = MetaUtils.getApplicationId();
    this.yarnUser = xdlUser;
    this.containerWorkDir = Utils.getCurDir();

    this.zookeeperConnStr = zookeeperConnStr;
    this.zookeeperRoot = zookeeperRoot;
    this.config = new YarnConfiguration();

    this.jobRoleName = containerJobName;
    this.jobIndex = jobIndex;

    this.volumeDirInHdfs = volumeDirInHdfs;
    this.needRemoveDockerRunner = needRemoveDockerRunner;
    this.cudaVisibleDevice = cudaVisibleDevice;
    this.cpuset = cpuset;

    LOG.info("Container workdir is:[{}]", this.containerWorkDir);
    LOG.info("Current xdl app name is: {}, config path: {}", this.yarnAppId, config);
    LOG.info("Creating XDLJobContainer:[{}] with index[{}], XDL app zookeeper address:[{}].", this.jobRoleName,
        this.jobIndex, this.zookeeperRoot);
  }

  public int run() throws Exception {
    if (!beforeRunCheck()) {
      return -1;
    }
    
    int exitCode = startXDL();
    for (int i = 0; i < schedulerConf.max_local_failover_times; i++) {
      if (this.killed) {
        break;
      }
      if (exitCode == 0) {
        break;
      }
      Utils.restartLog("Container local restart");
      exitCode = startXDL();
    }
    LOG.info("run finished status {}", exitCode);
    return exitCode;
  }

  public int getExitCode() {
    return this.exitCode;
  }

  protected String genCmd() throws Exception {
    String containerId = MetaUtils.getContainerId().toString();
    dockerName = String.format("xdl_%s_%s_%d_%s", this.yarnAppId, this.jobRoleName, this.jobIndex,
        containerId.substring(containerId.lastIndexOf('_') + 1));
    
    String imageName = null;
    String dockerRegistry = null;
    String registryUser = null;
    String registryPassword = null;

    if (this.schedulerConf.docker != null) {
      if (this.schedulerConf.docker.image != null) {
        imageName = this.schedulerConf.docker.image.trim();
      }

      if (this.schedulerConf.docker.registry != null) {
        dockerRegistry = this.schedulerConf.docker.registry.trim();
      }

      if (this.schedulerConf.docker.user != null) {
        registryUser = this.schedulerConf.docker.user;
      }

      if (this.schedulerConf.docker.password != null) {
        registryPassword = this.schedulerConf.docker.password;
      }
    }

    int cpuCores = 0;
    long memLimit = 0;
    long gpuCores = 0;
    String worker_script = null;

    String vp_method = "anneal";
    String meta_dir = null;
    if (this.schedulerConf.auto_rebalance != null) {
      if (this.schedulerConf.auto_rebalance.enable == true && this.schedulerConf.auto_rebalance.meta_dir != null) {
        vp_method = "balance";
        meta_dir = this.schedulerConf.auto_rebalance.meta_dir;
      }
    }
    if (cmdCode == 1) {
      vp_method = "anneal";
      if (this.schedulerConf.auto_rebalance.meta_dir != null) {
        meta_dir = this.schedulerConf.auto_rebalance.meta_dir;
      } else {
        String defaultFs = this.config.get("fs.defaultFS");
        String metaBase = MetaUtils.getMetaPath(this.config, this.yarnAppId);
        meta_dir = defaultFs + metaBase + "/meta_info/";
      }
    }
    if (cmdCode == 2) {
      vp_method = "balance";
      if (this.schedulerConf.auto_rebalance.meta_dir != null) {
        meta_dir = this.schedulerConf.auto_rebalance.meta_dir;
      } else {
        String defaultFs = this.config.get("fs.defaultFS");
        String metaBase = MetaUtils.getMetaPath(this.config, this.yarnAppId);
        meta_dir = defaultFs + metaBase + "/meta_info/";
      }
    }

    if (this.jobRoleName.equals(Constants.PS_JOB_NAME)) {
      cpuCores = this.schedulerConf.ps.cpu_cores;
      memLimit = (long) this.schedulerConf.ps.memory_m * 1024 * 1024;
      gpuCores = this.schedulerConf.ps.gpu_cores;
      worker_script = this.schedulerConf.script;
    } else if (this.jobRoleName.equals(Constants.WORKER_JOB_NAME)
        || this.jobRoleName.equals(Constants.SCHEDULER_JOB_NAME)) {
      cpuCores = this.schedulerConf.worker.cpu_cores;
      memLimit = (long) this.schedulerConf.worker.memory_m * 1024 * 1024;
      gpuCores = this.schedulerConf.worker.gpu_cores;
      worker_script = this.schedulerConf.script;
    } else {
      cpuCores = this.schedulerConf.extend_role.get(this.jobRoleName).cpu_cores;
      memLimit = (long) this.schedulerConf.extend_role.get(this.jobRoleName).memory_m * 1024 * 1024;
      gpuCores = this.schedulerConf.extend_role.get(this.jobRoleName).gpu_cores;
      worker_script = this.schedulerConf.extend_role.get(this.jobRoleName).script;
    }
    DockerCmdBuilder builder = DockerCmdBuilder.newBuilder().withUser(this.yarnUser).withScript(worker_script)
        .withParams("--config ../" + this.jobConfig).withJobName(this.jobRoleName).withTaskIndex(this.jobIndex)
        .withImageName(imageName).withDockerRegistry(dockerRegistry).withRegistryUser(registryUser).withRegistryPassword(registryPassword)
        .withGpuCores(gpuCores).withDockerContainerName(dockerName)
        .withEnvParams(this.schedulerConf.env_params).withVolumns(this.schedulerConf.volumns)
        .withLocalDirs(this.volumeDirInHdfs).withContainerWorkDir(this.containerWorkDir)
        .withCudaVisibleDevices(this.cudaVisibleDevice).withCpuCores(cpuCores).withMemoryLimit(memLimit)
        .withZkAddr("zfs://" + zookeeperConnStr + "/psplus/" + yarnAppId).withSchedulerConf(schedulerConf)
        .withAppId(yarnAppId).withVpMethod(vp_method).withMetaDir(meta_dir).withCpuSet(cpuset);

    String startCmd = builder.buildStartCmd();
    stopCmd = builder.buildStopCmd();
    dockerLogin = builder.buildDockerLoginCmd();
    pullImage = builder.buildPullImageCmd();
    removeCmd = builder.buildRemoveCmd();

    LOG.info("XDL docker container start cmd: {}; stop cmd: {} ", startCmd, stopCmd);

    return startCmd;
  }

  private int startXDL() throws Exception {
    String startCmd = genCmd();
    dockerLogin();
    pullImage();
    int exitCode = 0;
    try {
      processor = Runtime.getRuntime().exec(new String[] { "bash", "-c", startCmd });
      printStreamLog(processor);
      stoped = false;

      while (true) {
        try {
          exitCode = processor.exitValue();
          break;
        } catch (IllegalThreadStateException e) {
        }
        if(!checkJobState()) {
          return -1;
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
      LOG.error("Error occurs when wait docker finish.");
      return -1;
    } finally {
      stopXDL();
    }

    if (exitCode != 0) {
      LOG.error("Docker container run failed, Exit code is " + exitCode);
      return exitCode;
    }
    return 0;
  }

  protected volatile boolean stoped = false;
  private volatile boolean killed = false;

  protected synchronized void stopXDL() {
    if (stoped) {
      return;
    }
    stoped = true;
    long startTime = System.currentTimeMillis();
    if (processor != null) {
      LOG.info("stop container cmd [{}] ", stopCmd);
      exeCmd(stopCmd, startTime);
      LOG.info("stop container cost {} ms", System.currentTimeMillis() - startTime);

      if (needRemoveDockerRunner) {
        LOG.info("rm container cmd [{}] ", removeCmd);
        exeCmd(removeCmd, startTime);
        LOG.info("remove container cost {} ms", System.currentTimeMillis() - startTime);
      }

      try {
        processor.destroy();
      } catch (Exception e) {
        LOG.error("docker stop to destory processor!", e);
      }
      LOG.info("destroy processor cost {} ms", System.currentTimeMillis() - startTime);
    }
  }

  private synchronized void dockerLogin() {
    if (dockerLogin != null) {
      long startTime = System.currentTimeMillis();
      LOG.info("docker login [{}] ", dockerLogin);
      exeCmd(dockerLogin, startTime);
      LOG.info("docker login cost {} ms", System.currentTimeMillis() - startTime);
    }
  }

  private synchronized void pullImage() {
    long startTime = System.currentTimeMillis();
    LOG.info("pull image [{}] ", pullImage);
    exeCmd(pullImage, startTime);
    LOG.info("pull image cost {} ms", System.currentTimeMillis() - startTime);
  }

  protected void printStreamLog(final Process processor) {
    Thread std = new Thread(new Runnable() {
      @Override
      public void run() {
        BufferedReader in = new BufferedReader(new InputStreamReader(processor.getInputStream()));
        String line = null;
        try {
          while ((line = in.readLine()) != null && print_flag) {
            System.out.println(line);
          }
        } catch (IOException e) {
          e.printStackTrace();
        } finally {
          try {
            in.close();
          } catch (Exception e) {
            e.printStackTrace();
          }
        }
      }
    });
    std.setDaemon(true);
    Thread err = new Thread(new Runnable() {
      @Override
      public void run() {
        BufferedReader in = new BufferedReader(new InputStreamReader(processor.getErrorStream()));
        String line = null;
        try {
          while ((line = in.readLine()) != null && print_flag) {
            System.err.println(line);
          }
        } catch (IOException e) {
          e.printStackTrace();
        } finally {
          try {
            in.close();
          } catch (Exception e) {
            e.printStackTrace();
          }
        }
      }
    });
    err.setDaemon(true);
    std.start();
    err.start();
  }

  private void exeCmd(String cmd, long startTime) {
    int retryTimes = 3;
    while (retryTimes > 0) {
      retryTimes--;
      try {
        Process p = Runtime.getRuntime().exec(cmd);
        printStreamLog(p);
        int exitCode = p.waitFor();
        LOG.info("cmd [{}] cost {} ms status [{}] ", cmd, System.currentTimeMillis() - startTime, exitCode);
        if (exitCode == 0) {
          break;
        }
      } catch (Exception e) {
        LOG.error(cmd + " error!", e);
      }
    }
  }

  protected void close() {}

  public void stopBy(boolean killed) {
    if (killed) {
      this.killed = true;
    }
    stopXDL();
    this.close();
  }

  protected boolean beforeRunCheck() throws Exception {
    return true;
  }

  protected boolean checkJobState() throws Exception {
    return true;
  }
}
