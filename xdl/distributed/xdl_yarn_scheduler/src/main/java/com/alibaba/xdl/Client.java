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

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ApplicationReport;
import org.apache.hadoop.yarn.api.records.ApplicationSubmissionContext;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.URL;
import org.apache.hadoop.yarn.api.records.YarnApplicationState;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.client.api.YarnClientApplication;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.util.Apps;
import org.apache.hadoop.yarn.util.Records;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Client {

  private static Logger LOG = LoggerFactory.getLogger(Client.class);

  private static final int CHECK_APP_STATUS_INTERVAL = 100;

  private String config;
  private ArrayList<String> dependentFiles;
  private String appBasePath;
  private String volumes;
  private SchedulerConf schedulerConf;

  private ApplicationId appId = null;
  private YarnClient yarnClient = null;
  private Configuration conf = new YarnConfiguration();
  private String localTmpDir;

  public Client(String config, ArrayList<String> dependentFile, String uuid) {
    this.schedulerConf = Utils.parseSchedulerConf(config);
    this.volumes = "";
    this.config = config;
    this.dependentFiles = dependentFile;
    this.dependentFiles.addAll(Utils.splitString(schedulerConf.dependent_dirs, ","));
    this.localTmpDir = createLocalTmpDir(uuid);
  }

  private String createLocalTmpDir(String uuid) {
    String parDir = "/tmp/xdl_local";
    File parFile = new File(parDir);
    if (!parFile.exists()) {
      if (!parFile.mkdirs()) {
        throw new RuntimeException("Make dir failed: " + parDir);
      }
      if (!parFile.setWritable(true, false)) {
        throw new RuntimeException("Set dir writable failed: " + parDir);
      }
    }
    String dir = String.format("/tmp/xdl_local/%s", uuid);
    File file = new File(dir);
    if (!file.exists()) {
      if (!file.mkdirs()) {
        throw new RuntimeException("Make dir failed: " + dir);
      }
    }
    return dir;
  }

  public FinalApplicationStatus run() throws IOException, YarnException, InterruptedException {
    this.yarnClient = setupYarnClient(this.conf);

    YarnClientApplication app = yarnClient.createApplication();
    this.appId = app.getApplicationSubmissionContext().getApplicationId();
    LOG.info("Create application with id:[{}] success.", appId);

    this.appBasePath = Utils.genAppBasePath(this.conf, appId.toString(), Utils.getCurUser());
    LOG.info("Application base path:[{}].", appBasePath);
    this.uploadDependentFiles(appBasePath, this.dependentFiles);

    if (this.volumes.isEmpty()) {
      this.volumes = Constants.NULL_VALUE;
    }

    String amStartCommand = genCmd(this.config, this.volumes, Utils.getCurUser(), appBasePath);

    Map<String, LocalResource> resourceMap = setupResourceMap();

    LOG.info("local resources: {}", resourceMap);

    ContainerLaunchContext amContainer = setupApplicationMasterContainer(amStartCommand, resourceMap);
    ApplicationSubmissionContext appContext = setupApplicationMasterContext(amContainer, app);
    LOG.info("Submitting application {}", appId);
    yarnClient.submitApplication(appContext);

    FinalApplicationStatus appState = waitApplicationFinish(yarnClient, appId);

    return appState;
  }

  private Map<String, LocalResource> setupResourceMap() throws IOException {
    Map<String, LocalResource> resourceMap = new HashMap<String, LocalResource>();
    for (String file : this.dependentFiles) {
      if (file.endsWith(".jar")) {
        String fileName = Utils.extractFileName(file);
        LocalResource defConf = Records.newRecord(LocalResource.class);
        setupResource(new Path(this.appBasePath + fileName), defConf);
        resourceMap.put(fileName, defConf);
      }
    }
    {
      String fileName = Utils.extractFileName(config);
      LocalResource appXDLConfig = Records.newRecord(LocalResource.class);
      setupResource(new Path(this.appBasePath + fileName), appXDLConfig);
      resourceMap.put(fileName, appXDLConfig);
    }
    return resourceMap;
  }

  private ContainerLaunchContext setupApplicationMasterContainer(String amStartCommand,
      Map<String, LocalResource> resourceMap) {
    ContainerLaunchContext amContainer = Records.newRecord(ContainerLaunchContext.class);

    amContainer.setCommands(Collections.singletonList(amStartCommand));

    amContainer.setLocalResources(resourceMap);
    Map<String, String> appMasterEnv = setupAppMasterEnv();
    amContainer.setEnvironment(appMasterEnv);

    LOG.info("Setup ApplicationMaster container success.");

    return amContainer;
  }

  private ApplicationSubmissionContext setupApplicationMasterContext(ContainerLaunchContext amContainer,
      YarnClientApplication app) {

    Resource capability = Records.newRecord(Resource.class);
    capability.setMemorySize(Constants.APPLICATION_MASTER_MEMORY);
    capability.setVirtualCores(Constants.APPLICATION_MASTER_CORE);

    ApplicationSubmissionContext appContext = app.getApplicationSubmissionContext();
    appContext.setApplicationType(Constants.XDL_YARN_APP_TYPE);
    appContext.setApplicationName(this.schedulerConf.job_name);
    appContext.setAMContainerSpec(amContainer);
    appContext.setResource(capability);
    appContext.setQueue(this.schedulerConf.scheduler_queue);

    LOG.info("Setup application context success.");
    return appContext;
  }

  private FinalApplicationStatus waitApplicationFinish(YarnClient yarnClient, ApplicationId appId)
      throws InterruptedException, IOException, YarnException {

    ApplicationReport appReport = yarnClient.getApplicationReport(appId);
    YarnApplicationState appState = appReport.getYarnApplicationState();

    LOG.info("AppMaster host {} Start waiting application: {} ends.", appReport.getHost(), appId);
    while (appState != YarnApplicationState.FINISHED && appState != YarnApplicationState.KILLED
        && appState != YarnApplicationState.FAILED) {
      Thread.sleep(CHECK_APP_STATUS_INTERVAL);
      appReport = yarnClient.getApplicationReport(appId);
      appState = appReport.getYarnApplicationState();
    }
    LOG.info("Application {} finish with state {}", appId, appState);
    return appReport.getFinalApplicationStatus();
  }

  private YarnClient setupYarnClient(Configuration conf) {
    YarnClient yarnClient = YarnClient.createYarnClient();
    yarnClient.init(conf);
    yarnClient.start();
    LOG.info("Yarn client start success.");
    return yarnClient;
  }

  private void uploadDependentFiles(String basePath, ArrayList<String> dependentFileList) throws IOException {
    uploadLocalFileToHdfs(this.config, basePath);
    if (dependentFileList != null) {
      uploadFilesToHdfs(dependentFileList, basePath);
    }
    LOG.info("Upload user files success.");
  }

  private void uploadFilesToHdfs(ArrayList<String> localFiles, String destPath) throws IOException {
    LOG.info("begin to upload files to hdfs");
    Iterator<String> iter = localFiles.iterator();
    while (iter.hasNext()) {
      String fileName = iter.next();
      uploadLocalFileToHdfs(fileName, destPath);
    }
    LOG.info("finish uploading files to hdfs");
  }

  private String uploadLocalFileToHdfs(String srcFilePath, String dstHdfsDir) throws IOException {
    FileSystem fs = FileSystem.get(conf);
    File srcFile = new File(srcFilePath);
    if (srcFile.isDirectory()) {
      if (Files.isSymbolicLink(srcFile.toPath())) {
        java.nio.file.Path actualPath = Files.readSymbolicLink(srcFile.toPath());
        srcFile = actualPath.toFile();
      }
      String fileName = srcFile.getName();
      String dirName = srcFile.getParentFile().getAbsolutePath();
      String tarFileName = String.format("%s/%s.tar.gz", this.localTmpDir, fileName);
      Utils.runCmd(String.format("tar -czf %s -C %s ./%s", tarFileName, dirName, fileName));
      if (!volumes.isEmpty()) {
        volumes += Constants.OPTION_VALUE_SEPARATOR;
      }
      volumes += fileName + ".tar.gz";
      Path dstFilePath = new Path(dstHdfsDir + "/" + fileName + ".tar.gz");
      fs.copyFromLocalFile(new Path(tarFileName), dstFilePath);
      fs.close();
      LOG.info("Upload file {} to {} success.", srcFilePath, dstFilePath.toString());
      return dstFilePath.toString();
    } else {
      fs.copyFromLocalFile(new Path(srcFilePath), new Path(dstHdfsDir));
      fs.close();
      String fileName = Utils.extractFileName(srcFilePath);
      String dstFilePath = Paths.get(dstHdfsDir, fileName).toString();
      LOG.info("Upload file {} to {} success.", srcFilePath, dstFilePath);
      return dstFilePath;
    }
  }

  private String genCmd(String config, String volumes, String user, String basePath) {
    String master = " com.alibaba.xdl.AppMasterRunner";
    String command = "$JAVA_HOME/bin/java" + " -Xmx256M " + master + " -c=" + Utils.extractFileName(config) + " -v="
        + volumes + " -u=" + user + " -p=" + basePath + " 1>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout"
        + " 2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stderr";
    LOG.info("ApplicationMaster start command is: [{}]", command);
    return command;
  }

  private void setupResource(Path resourcePath, LocalResource localResource) throws IOException {
    FileStatus fileStatus;
    fileStatus = FileSystem.get(conf).getFileStatus(resourcePath);

    localResource.setResource(URL.fromPath(resourcePath));
    localResource.setSize(fileStatus.getLen());
    localResource.setTimestamp(fileStatus.getModificationTime());
    localResource.setType(LocalResourceType.FILE);
    localResource.setVisibility(LocalResourceVisibility.PUBLIC);
  }

  private Map<String, String> setupAppMasterEnv() {

    Map<String, String> appMasterEnv = new HashMap<String, String>();
    String classPathSeparator = System.getProperty("path.separator");
    for (String c : conf.getStrings(YarnConfiguration.YARN_APPLICATION_CLASSPATH,
        YarnConfiguration.DEFAULT_YARN_APPLICATION_CLASSPATH)) {
      LOG.info("Master add CLASSPATH:{}", c);
      Apps.addToEnvironment(appMasterEnv, Environment.CLASSPATH.name(), c.trim(), classPathSeparator);
    }

    Apps.addToEnvironment(appMasterEnv, Environment.CLASSPATH.name(), Environment.PWD.$() + File.separator + "*",
        classPathSeparator);
    return appMasterEnv;
  }

  public void dealWithInterruption() {
    try {
      YarnApplicationState appState = yarnClient.getApplicationReport(appId).getYarnApplicationState();
      if (appState == YarnApplicationState.RUNNING) {
        yarnClient.killApplication(appId);
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void clear() {
    try {
      Utils.removeHdfsDirs(conf, appBasePath);
    } catch (Exception e) {
    }
    try {
      File tmp = new File(this.localTmpDir);
      if (tmp.exists()) {
        FileUtils.deleteDirectory(tmp);
      }
    } catch (Exception e) {
    }
  }

  public static void main(String[] args) {
    CommandLine cmd = null;
    Options options = generateCLIOption();
    CommandLineParser parser = new GnuParser();
    try {
      cmd = parser.parse(options, args);
    } catch (ParseException e) {
      System.out.println(e.getMessage());
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("xdl client", options);
      System.exit(1);
    }

    ArrayList<String> dependentFileList = Utils.splitString(cmd.getOptionValue("dependent-files", ""),
        Constants.OPTION_VALUE_SEPARATOR);

    final Client client = new Client(cmd.getOptionValue("config"), dependentFileList, cmd.getOptionValue("uuid", ""));

    Runtime.getRuntime().addShutdownHook(new Thread() {
      @Override
      public void run() {
        client.dealWithInterruption();
        client.clear();
      }
    });

    boolean success = false;
    try {
      FinalApplicationStatus state = client.run();
      Utils.logFinalStatus(client.appId.toString(), state.toString());
      if (state == FinalApplicationStatus.SUCCEEDED) {
        success = true;
      }
    } catch (Exception e) {
      success = false;
      e.printStackTrace();
    } finally {
      client.clear();
    }
    if (!success) {
      System.exit(1);
    }
  }

  private static org.apache.commons.cli.Options generateCLIOption() {
    org.apache.commons.cli.Options options = new org.apache.commons.cli.Options();
    Option configPathOp = new Option("c", "config", true, "XDL application configuration path.");
    configPathOp.setRequired(true);
    options.addOption(configPathOp);

    Option fileToUpload = new Option("f", "dependent-files", true, "File to install in container work dir.");
    fileToUpload.setRequired(false);
    options.addOption(fileToUpload);

    Option uid = new Option("uuid", "uuid", true, "uuid");
    uid.setRequired(false);
    options.addOption(uid);

    return options;
  }
}
