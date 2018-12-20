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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import sun.misc.Signal;
import sun.misc.SignalHandler;

import com.alibaba.xdl.prerun.PreContainer;

@SuppressWarnings("restriction")
public class ContainerRunner {

  private static Logger LOG = LoggerFactory.getLogger(ContainerRunner.class);
  private static Configuration conf = new YarnConfiguration();

  public static void main(String[] args) throws Exception {
    Options options = generateCliOption();
    CommandLineParser parser = new GnuParser();
    CommandLine cmd = null;
    try {
      cmd = parser.parse(options, args);
    } catch (ParseException e) {
      System.out.println(e.getMessage());
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("container help", options);
      System.exit(1);
    }

    String containerJobName = cmd.getOptionValue("container_job_name");
    String xdlAppConfigPath = cmd.getOptionValue("xdl_app_config_path");
    int jobIndex = Integer.parseInt(cmd.getOptionValue("job_index"));
    String zookeeperConnStr = cmd.getOptionValue("zookeeper_conn_str");
    String zookeeperRoot = cmd.getOptionValue("zookeeper_root");
    String port = cmd.getOptionValue("xdl_port");
    String cudaDevice = cmd.getOptionValue("cuda_visible_device");
    String cpuset = cmd.getOptionValue("cpu_set");
    String xdlUser = cmd.getOptionValue("xdl_user", Constants.LINUX_ROOT_USER);
    String volumeDirInHdfs = cmd.getOptionValue("volume_dir_in_hdfs", Constants.NULL_VALUE);
    boolean needRemoveDockerContainer = Boolean.valueOf(cmd.getOptionValue("remove_docker_container", "true"));

    SchedulerConf jobConf = Utils.parseSchedulerConf(xdlAppConfigPath);
    boolean balance_enable = false;
    String meta_dir = null;
    if (jobConf.auto_rebalance != null) {
      balance_enable = jobConf.auto_rebalance.enable;
      meta_dir = jobConf.auto_rebalance.meta_dir;
    }

    ContainerBase container;
    if (balance_enable == false || Utils.existsHdfsFile(conf, meta_dir)) {
      container = new ContainerBase(xdlUser, containerJobName, xdlAppConfigPath, jobIndex, zookeeperConnStr,
          zookeeperRoot, needRemoveDockerContainer, volumeDirInHdfs, port, cudaDevice, cpuset);
    } else {
      container = new PreContainer(xdlUser, containerJobName, xdlAppConfigPath, jobIndex, zookeeperConnStr,
          zookeeperRoot, needRemoveDockerContainer, volumeDirInHdfs, port, cudaDevice, cpuset);
    }

    ContainerSignalHandler signalHandler = new ContainerSignalHandler(container);
    Signal.handle(new Signal("TERM"), signalHandler);
    Signal.handle(new Signal("INT"), signalHandler);
    try {
      int retCode = container.run();
      System.exit(retCode);
    } catch (Exception e) {
      LOG.error("run docker error!", e);
      System.exit(-1);
    } finally {
      container.stopBy(false);
    }
  }

  private static Options generateCliOption() {
    Options options = new Options();

    Option jobNameOp = new Option("j", "container_job_name", true, "Job name of xdl:[worker/ps]");
    jobNameOp.setRequired(true);
    options.addOption(jobNameOp);

    Option configPathOp = new Option("c", "xdl_app_config_path", true, "XDL application configuration path.");
    configPathOp.setRequired(true);
    options.addOption(configPathOp);

    Option jobIndexOp = new Option("i", "job_index", true, "Job index of xdl application.");
    jobIndexOp.setRequired(true);
    options.addOption(jobIndexOp);

    Option zookeeperConnStr = new Option("z", "zookeeper_conn_str", true, "Connect string for zookeeper");
    zookeeperConnStr.setRequired(true);
    options.addOption(zookeeperConnStr);

    Option zookeeperRoot = new Option("r", "zookeeper_root", true, "Root node for xdl application");
    zookeeperRoot.setRequired(true);
    options.addOption(zookeeperRoot);

    Option xdlPort = new Option("xp", "xdl_port", true, "xdl port from xdl program");
    xdlPort.setRequired(false);
    options.addOption(xdlPort);

    Option dockerHost = new Option("h", "docker_host", true, "Docker host");
    dockerHost.setRequired(false);
    options.addOption(dockerHost);

    Option cd = new Option("cd", "cuda_visible_device", true, "Cuda visible devices");
    cd.setRequired(false);
    options.addOption(cd);
    
    Option cs = new Option("cpuset", "cpu_set", true, "cpu set");
    cs.setRequired(false);
    options.addOption(cs);

    Option xdlUser = new Option("u", "xdl_user", true, "user for this app");
    xdlUser.setRequired(false);
    options.addOption(xdlUser);

    Option volumeDirInHdfs = new Option("v", "volume_dir_in_hdfs", true, "user for this app");
    volumeDirInHdfs.setRequired(false);
    options.addOption(volumeDirInHdfs);

    Option removeDockerContainer = new Option("d", "remove_docker_container", true,
        "Whether remove docker container after task finished.");
    removeDockerContainer.setRequired(false);
    options.addOption(removeDockerContainer);
    return options;
  }

  private static class ContainerSignalHandler implements SignalHandler {
    private static Logger LOG = LoggerFactory.getLogger(ContainerSignalHandler.class);

    private ContainerBase jobContainer = null;

    public ContainerSignalHandler(ContainerBase jobContainer) {
      this.jobContainer = jobContainer;
    }

    public void handle(Signal signal) {
      LOG.info("Container is killed by signal:{}.", signal.getNumber());
      this.jobContainer.stopBy(true);
    }
  }
}
