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
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.alibaba.xdl.prerun.PreAppMaster;
import com.alibaba.xdl.SchedulerConf;
import com.alibaba.xdl.Utils;
import com.alibaba.xdl.meta.Meta.Status;

import sun.misc.Signal;
import sun.misc.SignalHandler;

@SuppressWarnings("restriction")
public class AppMasterRunner {

  private static Logger LOG = LoggerFactory.getLogger(AppMasterRunner.class);
  private static Configuration conf = new YarnConfiguration();

  public static void main(String[] args) throws Exception {
    org.apache.commons.cli.Options options = generateCLIOption();
    CommandLineParser parser = new GnuParser();
    CommandLine cmd = null;
    try {
      cmd = parser.parse(options, args);
    } catch (ParseException e) {
      System.out.println(e.getMessage());
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("xdl-on-yarn-applicationMaster", options);
      System.exit(1);
    }

    String config = cmd.getOptionValue("config");
    String basePath = cmd.getOptionValue("base-path");
    String user = cmd.getOptionValue("user");
    String volumes = cmd.getOptionValue("volume");

    SchedulerConf jobConf = Utils.parseSchedulerConf(config);
    boolean balance_enable = false;
    String meta_dir = null;
    if (jobConf.auto_rebalance != null) {
      balance_enable = jobConf.auto_rebalance.enable;
      meta_dir = jobConf.auto_rebalance.meta_dir;
    }

    AppMasterBase applicationMaster;
    if (balance_enable == false || Utils.existsHdfsFile(conf, meta_dir)) {
      applicationMaster = new AppMasterBase(basePath, user, config, volumes);
    } else {
      applicationMaster = new PreAppMaster(basePath, user, config, volumes);
    }

    ApplicationMasterSignalHandler signalHandler = new ApplicationMasterSignalHandler(applicationMaster);
    Signal.handle(new Signal("TERM"), signalHandler);
    Signal.handle(new Signal("INT"), signalHandler);

    try {
      Status status = applicationMaster.run();
      applicationMaster.dealWithExit(status);
    } catch (Exception e) {
      LOG.error("run error!", e);
      applicationMaster.dealWithExit(Status.FAIL);
    }
  }

  private static org.apache.commons.cli.Options generateCLIOption() {
    org.apache.commons.cli.Options options = new org.apache.commons.cli.Options();
    Option appStageDir = new Option("p", "base-path", true, "App stage hdfs dir to store config and scripts.");
    appStageDir.setRequired(true);
    options.addOption(appStageDir);

    Option appConfig = new Option("c", "config", true, "HDFS path of jar for xdl app config.");
    appConfig.setRequired(true);
    options.addOption(appConfig);

    Option xdlUser = new Option("u", "user", true, "user for this app.");
    xdlUser.setRequired(true);
    options.addOption(xdlUser);

    Option volumeDirInHdfs = new Option("v", "volume", true, "volume in hdfs to bind in container.");
    volumeDirInHdfs.setRequired(false);
    options.addOption(volumeDirInHdfs);

    return options;
  }

  private static class ApplicationMasterSignalHandler implements SignalHandler {
    private static Logger LOG = LoggerFactory.getLogger(ApplicationMasterSignalHandler.class);
    private AppMasterBase applicationMaster = null;

    public ApplicationMasterSignalHandler(AppMasterBase applicationMaster) {
      this.applicationMaster = applicationMaster;
    }

    @Override
    public void handle(Signal signal) {
      LOG.info("Application Master is killed by signal:{}", signal.getNumber());
      applicationMaster.dealWithExit(Status.KILLED);
    }
  }
}
