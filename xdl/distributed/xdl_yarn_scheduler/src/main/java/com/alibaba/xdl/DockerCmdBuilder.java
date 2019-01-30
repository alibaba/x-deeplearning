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
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DockerCmdBuilder {

  private static Logger LOG = LoggerFactory.getLogger(DockerCmdBuilder.class);

  private String user;
  private String workDir;
  private String pythonDir;
  private String cudaVisibleDevice = Constants.NULL_VALUE;
  private String cpuset = Constants.NULL_VALUE;

  private String script;
  private String params;
  private String jobName;
  private int taskIndex;
  private String startCommand;
  private String zkAddr;
  private String appId;
  private final String corefileCommand = "ulimit -c unlimited";
  private final String envCommand = "source /etc/profile";

  private String dockerContainerName;
  private String imageName;
  private String localDirs;
  private String containerWorkDir;
  private List<String> volumeDirs;
  private int cpuCores;
  private long gpuCores;
  private long memoryLimit;
  private SchedulerConf schedulerConf;
  private String vp_method;
  private String meta_dir;

  public DockerCmdBuilder withVpMethod(String vp_method) {
    this.vp_method = vp_method;
    return this;
  }

  public DockerCmdBuilder withMetaDir(String meta_dir) {
    this.meta_dir = meta_dir;
    return this;
  }

  private DockerCmdBuilder() {
  }

  public static DockerCmdBuilder newBuilder() {
    return new DockerCmdBuilder();
  }

  public DockerCmdBuilder withUser(String user) {
    this.user = user;
    if (this.user.equals(Constants.LINUX_ROOT_USER)) {
      this.workDir = Constants.XDL_WORK_DIR_IN_DOCKER;
      this.pythonDir = Constants.XDL_PYTHON_DIR_IN_DOCKER;
    } else {
      this.workDir = String.format(Constants.XDL_WORK_DIR_PATTERN_IN_DOCKER, user);
      this.pythonDir = String.format(Constants.XDL_PYTHON_DIR_PATTERN_IN_DOCKER, user);
    }
    return this;
  }

  public DockerCmdBuilder withScript(String script) {
    this.script = script;
    return this;
  }

  public DockerCmdBuilder withParams(String... params) {
    StringBuilder str = new StringBuilder();
    for (String p : params) {
      if (StringUtils.isNotBlank(p)) {
        str.append(p).append(" ");
      }
    }
    this.params = str.toString();
    return this;
  }

  public DockerCmdBuilder withJobName(String jobName) {
    this.jobName = jobName;
    return this;
  }

  public DockerCmdBuilder withZkAddr(String zkAddr) {
    this.zkAddr = zkAddr;
    return this;
  }

  public DockerCmdBuilder withAppId(String appId) {
    this.appId = appId;
    return this;
  }

  public DockerCmdBuilder withTaskIndex(int taskIndex) {
    this.taskIndex = taskIndex;
    return this;
  }

  public DockerCmdBuilder withStartCommand(String startCommand) {
    this.startCommand = startCommand;
    return this;
  }

  public DockerCmdBuilder withCudaVisibleDevices(String cudaVisibleDevice) {
    this.cudaVisibleDevice = cudaVisibleDevice;
    return this;
  }
  
  public DockerCmdBuilder withCpuSet(String cpuset) {
    this.cpuset = cpuset;
    return this;
  }

  public String getWorkDir() {
    if (StringUtils.isNotBlank(schedulerConf.dependent_dirs)) {
      String firstDir = schedulerConf.dependent_dirs.split(",")[0];
      if (StringUtils.isNotBlank(firstDir)) {
        firstDir = Utils.extractFileName(firstDir);
        return workDir + "/" + firstDir;
      }
    }
    return workDir;
  }

  public String getDockerImage() {
    return imageName;
  }

  public DockerCmdBuilder withDockerContainerName(String dockerContainerName) {
    this.dockerContainerName = dockerContainerName;
    return this;
  }

  public DockerCmdBuilder withImageName(String imageName) {
    this.imageName = imageName;
    return this;
  }

  public DockerCmdBuilder withLocalDirs(String localDirs) {
    this.localDirs = localDirs;
    return this;
  }

  public DockerCmdBuilder withContainerWorkDir(String containerWorkDir) throws IOException {
    this.containerWorkDir = containerWorkDir;
    this.volumeDirs = getSymbolicDirs(containerWorkDir, listFiles(containerWorkDir));
    return this;
  }

  public DockerCmdBuilder withCpuCores(int cpuCores) {
    this.cpuCores = cpuCores;
    return this;
  }

  public DockerCmdBuilder withGpuCores(long gpuCores) {
    this.gpuCores = gpuCores;
    return this;
  }

  public DockerCmdBuilder withMemoryLimit(long memoryLimit) {
    this.memoryLimit = memoryLimit;
    return this;
  }

  public DockerCmdBuilder withSchedulerConf(SchedulerConf schedulerConf) {
    this.schedulerConf = schedulerConf;
    return this;
  }

  private List<String> listFiles(String containerWorkDir) {
    List<String> fileList = new ArrayList<String>();
    File[] files;
    files = new File(containerWorkDir).listFiles();
    if (files == null) {
      return Collections.emptyList();
    }
    for (File file : files) {
      fileList.add(file.getName());
    }
    return fileList;
  }

  private List<String> getSymbolicDirs(String containerWorkDir, List<String> files) throws IOException {
    List<String> symbolicDirs = new ArrayList<String>();
    for (String fileName : files) {
      Path path = new File(String.format("%s/%s", containerWorkDir, fileName)).toPath();
      if (Files.isSymbolicLink(path)) {
        Path actualPath = Files.readSymbolicLink(path);
        LOG.info("{} is symbolic link, actual path is: {}", path, actualPath);
        String dir = actualPath.getParent().toString();
        LOG.info("Add {} to need volume dir list.", dir);
        symbolicDirs.add(dir);
      }
    }
    return symbolicDirs;
  }

  public String buildStartCmd() {
    buildPythonPath();
    String startCommand = buildStartCommand();

    String docker = "nvidia-docker run ";
    String line = "";
    try {
      if (gpuCores > 0 && !"scheduler".equals(jobName)) {
        Runtime rt = Runtime.getRuntime();
        String nvidiaCmd = "curl -s http://localhost:3476/docker/cli";
        Process pr = rt.exec(nvidiaCmd);
        BufferedReader input = new BufferedReader(new InputStreamReader(pr.getInputStream()));
        line = input.readLine();
        int exitVal = pr.waitFor();
        if (exitVal != 0) {
          line = "";
          LOG.error("Fail to execute " + nvidiaCmd);
        }
      }
    } catch (Exception e) {
      LOG.error(e.toString());
      line = "";
    }
    docker = "nvidia-docker run " + line + " ";
    StringBuilder cmd = new StringBuilder(docker);
    cmd.append(" --expose=").append(Constants.DOCKER_DEFAULT_PORT);
    cmd.append(" -m=").append(memoryLimit);
    cmd.append(" -w=").append(getWorkDir());
    cmd.append(" -t ");
    cmd.append(" --name=").append(dockerContainerName);
    if (this.schedulerConf.bind_cores == true && !cpuset.equals(Constants.NULL_VALUE)) {
      cmd.append(" --cpuset-cpus=" + cpuset);
    } else {
      cmd.append(" --cpu-period=").append(Constants.CPU_PERIOD_BASE);
      cmd.append(" --cpu-quota=").append(this.cpuCores * Constants.CPU_PERIOD_BASE);
      cmd.append(" -c ").append(this.cpuCores);
    }
    cmd.append(" --net=").append(Constants.DOCKER_NETWORK_MODE);
    cmd.append(" -e ").append(String.format("%s=%s", Constants.PYTHON_PATH_ENV_NAME, this.pythonPath));
    if (!cudaVisibleDevice.equals(Constants.NULL_VALUE)) {
      cmd.append(" -e ").append(String.format("%s=%s", "CUDA_VISIBLE_DEVICES", cudaVisibleDevice));
    }
    cmd.append(" -e ").append(String.format("%s=%s", "vp_method", this.vp_method));
    if (this.meta_dir != null) {
      cmd.append(" -e ").append(String.format("%s=%s", "meta_dir", this.meta_dir));
    }
    cmd.append(" --entrypoint=bash");
    bindVolumes(cmd);
    cmd.append(" ").append(this.getDockerImage());
    cmd.append(" -c ").append(String.format("'%s && %s && %s'", this.corefileCommand, this.envCommand, startCommand));
    return cmd.toString().replaceAll("-e CUDA_VISIBLE_DEVICES=GPU_LIST_PLACEHOLDER", " ");
  }

  public String buildStopCmd() {
    String docker = "docker stop -t 30 ";
    return new StringBuilder(docker).append(this.dockerContainerName).toString();
  }

  public String buildPullImageCmd() {
    String docker = "docker pull ";
    return new StringBuilder(docker).append(this.getDockerImage()).toString();
  }

  public String buildRemoveCmd() {
    String docker = "docker rm ";
    return new StringBuilder(docker).append(this.dockerContainerName).toString();
  }

  public String buildStatsCmd() {
    String docker = "docker stats --no-stream=true ";
    return new StringBuilder(docker).append(this.dockerContainerName).toString();
  }

  private void bindVolumes(StringBuilder cmd) {
    bindVolume(cmd, containerWorkDir, workDir, null);
    if (volumeDirs != null) {
      for (String dir : volumeDirs) {
        if (containTar(dir)) {
          bindTarVolume(dir, cmd);
        } else {
          bindVolume(cmd, dir, dir, null);
        }
      }
    }
  }

  private void bindVolume(StringBuilder cmd, String from, String to, String mode) {
    cmd.append(" -v=").append(from).append(":").append(to);
    if (StringUtils.isNotBlank(mode)) {
      cmd.append(":").append(mode);
    }
  }

  private boolean containTar(String dir) {
    String[] tarFiles = localDirs.split(Constants.OPTION_VALUE_SEPARATOR);
    String file = ((new File(dir)).listFiles())[0].getName();
    LOG.info("volumeDirInLocal is {}, file is {}", localDirs, file);
    for (String tarFile : tarFiles) {
      if (tarFile.equals(file)) {
        return true;
      }
    }
    return false;
  }

  private void bindTarVolume(String dir, StringBuilder cmd) {
    LOG.info("Binding {} in bindTarVolume", dir);
    String file = ((new File(dir)).listFiles())[0].getName();
    String fileWithoutSuffix = file.replace(".tar.gz", "");
    this.pythonPath += ":" + fileWithoutSuffix;
    bindVolume(cmd, String.format("%s/%s/%s", dir, file, fileWithoutSuffix),
        String.format("%s/%s", this.workDir, fileWithoutSuffix), "rw");
  }

  private String pythonPath;

  private void buildPythonPath() {
    this.pythonPath = String.format("%s:%s", this.pythonDir, this.workDir);
    String[] tarFiles = this.localDirs.split(Constants.OPTION_VALUE_SEPARATOR);
    StringBuilder sb = new StringBuilder();
    for (String tarFile : tarFiles) {
      if (tarFile.endsWith(".tar.gz")) {
        if (sb.length() != 0) {
          sb.append(":");
        }
        sb.append(workDir).append("/").append(tarFile.replace(".tar.gz", ""));
      }
    }
    if (sb.length() != 0) {
      this.pythonPath += ":" + sb.toString();
    }
  }

  public String writeLaunchScript() {
    return script;
  }

  private String buildStartCommand() {
    String _script = writeLaunchScript();
    if (startCommand == null) {
      StringBuilder cmd = new StringBuilder();
      cmd.append("python ").append(_script).append(" --task_name ").append(jobName).append(" --task_index ")
          .append(taskIndex).append(" --run_mode dist ").append(" --zk_addr ").append(zkAddr).append(" --app_id ")
          .append(appId).append(" ").append(params);

      startCommand = cmd.toString();
    } else {
      startCommand = "";
    }
    if (!user.equals(Constants.LINUX_ROOT_USER)) {
      return wrapWithAddNewUserCommand();
    }
    return startCommand;
  }

  private String wrapWithAddNewUserCommand() {
    StringBuilder envBuilder = new StringBuilder();
    envBuilder.append(String.format("%s=%s ", Constants.PYTHON_PATH_ENV_NAME, this.pythonPath));
    envBuilder.append(" JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64 ");
    envBuilder.append(" HADOOP_HOME=/usr/lib/hadoop-2.6.0/ ");
    envBuilder.append(" HADOOP_HDFS_HOME=$HADOOP_HOME ");
    envBuilder.append(" LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64:$JAVA_HOME/jre/lib/amd64/server/ ");
    envBuilder.append(" PATH=$HADOOP_HOME/bin/:$PATH ");
    envBuilder.append(" HADOOP_CONF_DIR=/usr/lib/hadoop-2.6.0/etc/hadoop/ ");
    return startCommand;
  }
}
