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
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.retry.RetryOneTime;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.protocolrecords.AllocateResponse;
import org.apache.hadoop.yarn.api.protocolrecords.RegisterApplicationMasterResponse;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.URL;
import org.apache.hadoop.yarn.client.api.AMRMClient;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.client.api.NMClient;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.util.Apps;
import org.apache.hadoop.yarn.util.Records;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.alibaba.fastjson.JSON;
import com.alibaba.xdl.SchedulerConf.ExtendRoleResource;
import com.alibaba.xdl.meta.MetaHandler;
import com.alibaba.xdl.meta.MetaUtils;
import com.alibaba.xdl.meta.Meta.ContainerMeta;
import com.alibaba.xdl.meta.Meta.Status;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

public class AppMasterBase {

  protected static Logger LOG = LoggerFactory.getLogger(AppMasterBase.class);

  protected String appId;
  private float responseId = 0.01f;
  protected Configuration yarnConf = null;
  private String zookeeperConnStr = null;
  protected SchedulerConf schedulerConf;
  protected JobConf jobConf;

  private String xdlUser = null;
  private String config = null;
  private Map<String, String> containerEnv = null;
  private String volumes = "";

  private String basePath = null;
  private Map<String, LocalResource> localResources = null;

  protected int workNumber = 10;
  protected int psNumber = 2;
  private int schedulerNum = 0;

  private Resource workerContainerCapability = null;
  private Resource psContainerCapability = null;
  private Resource schedulerCapability = null;

  private Map<String, Integer> extendRoleNum = null;
  private Map<String, Resource> extendRoleContainerCapability = null;
  private Map<String, Container[]> launchExtendRoleContainers = null;
  private Map<String, Map<ContainerId, Integer>> extendRoleContainer = null;

  private Container[] launchWorkerContainers = null;
  private Container[] launchPsContainers = null;
  private Map<ContainerId, Integer> workContainer = null;
  private Map<ContainerId, Integer> psContainer = null;
  private Container scheduler;

  private MetaHandler metaHandler;

  private CuratorFramework zkCli;

  public AppMasterBase(String basePath, String user, String config, String volumes) {
    this.schedulerConf = Utils.parseSchedulerConf(config);
    this.yarnConf = new YarnConfiguration();
    this.basePath = basePath;
    this.config = config;
    this.xdlUser = user;
    this.volumes = volumes;

    schedulerNum = 1;

    this.workerContainerCapability = Records.newRecord(Resource.class);
    this.psContainerCapability = Records.newRecord(Resource.class);
    this.schedulerCapability = Records.newRecord(Resource.class);
    this.workContainer = Maps.newHashMap();
    this.psContainer = Maps.newHashMap();

    this.extendRoleContainerCapability = Maps.newHashMap();
    this.extendRoleNum = Maps.newHashMap();
    this.launchExtendRoleContainers = Maps.newHashMap();
    this.extendRoleContainer = Maps.newHashMap();

    this.appId = MetaUtils.getApplicationId();

    this.zookeeperConnStr = yarnConf.get("yarn.resourcemanager.zk-address");
    LOG.info("Zookeeper connect string is:[{}]", this.zookeeperConnStr);

    zkCli = CuratorFrameworkFactory.newClient(zookeeperConnStr, new RetryOneTime(2000));
    zkCli.start();
  }

  private void updateJobConf() throws Exception {
    this.jobConf = Utils.parseJobConf(config);

    String fileName = config;
    int subIndex = config.lastIndexOf('/');
    if (subIndex >= 0) {
      fileName = config.substring(subIndex + 1);
    }
    Utils.writeHdfsFile(yarnConf, this.basePath, fileName, JSON.toJSONString(this.jobConf.getJobConf()));
  }

  public Status run() throws Exception {
    this.updateJobConf();

    this.createRMClient();
    this.createNMClient();

    this.init();
    this.initResource();

    metaHandler = MetaHandler.newInstance(psNumber, workNumber, schedulerNum, yarnConf);
    metaHandler.flush();
    metaHandler.updateDirs(this.jobConf.getCheckpointOutputDir(), this.jobConf.getSummaryOutputDir());

    long startTime = System.currentTimeMillis();
    this.requestSchedulerContainers(startTime, schedulerNum);
    this.priority++;
    this.requestPsContainers(startTime, psNumber);
    this.priority++;
    this.requestWorkerContainers(startTime, workNumber);
    this.priority++;
    this.requestExtendRoleContainers(startTime, extendRoleNum);
    this.priority++;
    launchScheduler(nmClient);
    launchPs(nmClient);
    launchWorker(nmClient);
    launchExtendRole(nmClient);

    metaHandler.flush();

    return waitForWorkerFinish(rmClient, nmClient) ? Status.FINISHED : Status.FAIL;
  }

  private void launchContainer(NMClient nmClient, Container container, String workType, int workIndex, String port,
      FailoverTimes times) throws Exception {
    ContainerLaunchContext ctx = Records.newRecord(ContainerLaunchContext.class);
    if (volumes.isEmpty()) {
      volumes = Constants.NULL_VALUE;
    }
    String command = genCmd(this.xdlUser, workType, workIndex, this.config, Constants.ZOO_KEEPER_PATH_ROOT,
        zookeeperConnStr, this.volumes, port);
    ctx.setCommands(Collections.singletonList(command));
    LOG.info("Launching {} container [{}]", workType, container.getId());

    ctx.setLocalResources(this.localResources);
    ctx.setEnvironment(this.containerEnv);

    nmClient.startContainer(container, ctx);
    LOG.info("container Id is {},node Id is {}", container.getId(), container.getNodeId());

    metaHandler.addContainer(newContainerMeta(container, workType, workIndex, times, command, port));
  }

  private String genCmd(String xdlUser, String workType, int workIndex, String config, String xdlZKRoot,
      String zookeeperConnStr, String volumeDirInHdfs, String port) {
    String container = " com.alibaba.xdl.ContainerRunner";
    String command = "$JAVA_HOME/bin/java" + " -Xmx256M" + container + " -c=" + config + " -j=" + workType + " -i="
        + Integer.toString(workIndex) + " -z=" + zookeeperConnStr + " -r=" + xdlZKRoot + " -u=" + xdlUser + " -v="
        + volumeDirInHdfs + " -cpuset=_CPU_LIST_" + " -cd=" + "GPU_LIST_PLACEHOLDER" + (port == null ? "" : (" -xp=" + port)) 
        + " 1> " + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout" + " 2> " 
        + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stderr";
    LOG.info("container start command is {} ", command);
    return command;
  }

  private ContainerMeta newContainerMeta(Container container, String role, int index, FailoverTimes times, String cmd,
      String port) {
    ContainerMeta cm = new ContainerMeta();
    cm.containerId = container.getId().toString();
    cm.hostName = container.getNodeId().getHost();
    cm.port = port;
    cm.taskIdx = index;
    cm.status = Status.RUNNING;
    if (times != null) {
      cm.curAttamptIdx = times.getFailTimes();
    }

    cm.role = role;
    cm.startTime = new Date();
    cm.cores = container.getResource().getVirtualCores();
    cm.memoryMB = container.getResource().getMemorySize();
    cm.command = cmd;
    String logDirs = System.getenv(ApplicationConstants.Environment.LOG_DIRS.toString());
    logDirs = logDirs.substring(0, logDirs.lastIndexOf('/'));
    cm.stderr = logDirs + "/" + cm.containerId + "/stderr";
    cm.stdout = logDirs + "/" + cm.containerId + "/stdout";
    return cm;
  }

  private void launchScheduler(NMClient nmClient) throws Exception {
    if (scheduler != null) {
      launchContainer(nmClient, scheduler, Constants.SCHEDULER_JOB_NAME, 0, null, null);
    }
  }

  private void launchPs(NMClient nmClient) throws Exception {
    for (int i = 0; i < psNumber; i++) {
      Container container = launchPsContainers[i];
      if (container != null) {
        psContainer.put(container.getId(), i);
        launchContainer(nmClient, container, Constants.PS_JOB_NAME, i, null, null);
      }
    }
  }

  private void launchWorker(NMClient nmClient) throws Exception {
    for (int i = 0; i < workNumber; i++) {
      Container container = launchWorkerContainers[i];
      if (container != null) {
        workContainer.put(container.getId(), i);
        launchContainer(nmClient, container, Constants.WORKER_JOB_NAME, i, null, null);
      }
    }
  }

  private void launchExtendRole(NMClient nmClient) throws Exception {
    Iterator<Map.Entry<String, Integer>> iter = extendRoleNum.entrySet().iterator();
    while (iter.hasNext()) {
      Map.Entry<String, Integer> entry = iter.next();
      String work_type = entry.getKey();
      int work_num = entry.getValue();
      Map<ContainerId, Integer> containerMap = Maps.newHashMap();
      for (int i = 0; i < work_num; i++) {
        Container container = launchExtendRoleContainers.get(work_type)[i];
        if (container != null) {
          containerMap.put(container.getId(), i);
          launchContainer(nmClient, container, work_type, i, null, null);
        }
      }
      extendRoleContainer.put(work_type, containerMap);
    }
  }

  private int succeedWorkContainers = 0;
  private int containerFailedTime = 0;

  private boolean waitForWorkerFinish(AMRMClient<ContainerRequest> rmClient, NMClient nmClient) throws Exception {
    int total_workNumber = workNumber;
    Iterator<Map.Entry<String, Integer>> iter = extendRoleNum.entrySet().iterator();
    while (iter.hasNext()) {
      Map.Entry<String, Integer> entry = iter.next();
      String work_type = entry.getKey();
      if (this.schedulerConf.extend_role.get(work_type).xdl_worker == true) {
        int work_num = entry.getValue();
        total_workNumber += work_num;
      }
    }

    float deltaResponse = 1.0f / total_workNumber;
    while (succeedWorkContainers < total_workNumber) {
      this.preRun();
      this.processResponse(rmClient.allocate(responseId).getCompletedContainersStatuses(), nmClient);
      this.launchForFailover();
      responseId = deltaResponse * succeedWorkContainers;
      if (this.schedulerConf.min_finish_worker_rate < 100) {
        if (responseId * 100 >= this.schedulerConf.min_finish_worker_rate) {
          return true;
        }
      } else {
        int maxFinishNum = this.schedulerConf.min_finish_worker_num;
        if (this.schedulerConf.min_finish_worker_num == 0) {
          maxFinishNum = total_workNumber;
        }
        if (maxFinishNum <= succeedWorkContainers) {
          return true;
        }
      }
      Thread.sleep(100);
    }
    return true;
  }

  private void processResponse(List<ContainerStatus> completeDContainers, NMClient nmClient) throws Exception {
    if (completeDContainers.size() == 0) {
      return;
    }
    LOG.info("response container size {}", completeDContainers.size());
    for (ContainerStatus status : completeDContainers) {
      LOG.info("Completed container {} finish state is {} exit status {}", status.getContainerId(), status.getState(),
          status.getExitStatus());
      if (status.getExitStatus() == 0) {
        metaHandler.updateContainer(status.getContainerId().toString(), Status.FINISHED);

        ContainerId containerId = status.getContainerId();
        if (workContainer.get(containerId) != null) {
          succeedWorkContainers += 1;
          LOG.info("{} containers has successfully completed ", succeedWorkContainers);
          launchWorkerContainers[workContainer.get(containerId)] = null;
        }

        Iterator<Map.Entry<String, Map<ContainerId, Integer>>> iter = extendRoleContainer.entrySet().iterator();
        while (iter.hasNext()) {
          Map.Entry<String, Map<ContainerId, Integer>> entry = iter.next();
          String work_type = entry.getKey();
          if (this.schedulerConf.extend_role.get(work_type).xdl_worker == true) {
            Map<ContainerId, Integer> containerMap = entry.getValue();
            if (containerMap.get(containerId) != null) {
              succeedWorkContainers += 1;
              LOG.info("{} containers has successfully completed ", succeedWorkContainers);
              launchExtendRoleContainers.get(work_type)[containerMap.get(containerId)] = null;
            }
          }
        }
        rmClient.releaseAssignedContainer(status.getContainerId());
      } else {
        ContainerId containerId = status.getContainerId();
        if (workContainer.get(containerId) != null) {
          LOG.info("{} work container lost, lose exit status is {}, Launch it again", containerId,
              status.getExitStatus());
          int workIndex = workContainer.get(containerId);
          requestFailoverNodes(launchWorkerContainers[workIndex], workIndex, Constants.WORKER_JOB_NAME);
          launchWorkerContainers[workIndex] = null;
        } else if (psContainer.get(containerId) != null) {
          LOG.info("{} ps container lost, lose exit status is {}, Launch it again", containerId,
              status.getExitStatus());
          int psIndex = psContainer.get(containerId);
          requestFailoverNodes(launchPsContainers[psIndex], psIndex, Constants.PS_JOB_NAME);
          launchPsContainers[psIndex] = null;
        } else if (scheduler != null && scheduler.getId().equals(containerId)) {
          LOG.info("{} scheduler container lost, lose exit status is {}, Launch it again", containerId,
              status.getExitStatus());
          requestFailoverNodes(scheduler, 0, Constants.SCHEDULER_JOB_NAME);
          scheduler = null;
        } else {
          boolean flag = true;
          Iterator<Map.Entry<String, Map<ContainerId, Integer>>> iter = extendRoleContainer.entrySet().iterator();
          while (iter.hasNext()) {
            Map.Entry<String, Map<ContainerId, Integer>> entry = iter.next();
            String work_type = entry.getKey();
            Map<ContainerId, Integer> containerMap = entry.getValue();
            if (containerMap.get(containerId) != null) {
              LOG.info("{} {} container lost, lose exit status is {}, Launch it again", containerId, work_type,
                  status.getExitStatus());
              int roleIndex = containerMap.get(containerId);
              requestFailoverNodes(launchExtendRoleContainers.get(work_type)[roleIndex], roleIndex, work_type);
              launchExtendRoleContainers.get(work_type)[roleIndex] = null;
              flag = false;
            }
          }
          if (flag == true) {
            LOG.info("released conatiner {} status {}", containerId.toString(), status.getExitStatus());
            continue;
          }
        }
        containerFailedTime += 1;
        if (containerFailedTime > schedulerConf.max_failover_times) {
          LOG.info("container has failed {} times, shutdown this application", containerFailedTime);
          throw new RuntimeException(
              "container has failed " + containerFailedTime + " times,shutdown this application");
        }
      }
    }

    metaHandler.flush();
  }

  private List<Request> failoverRequests = Lists.newLinkedList();
  private List<Request> failoverLaunchRequests = Lists.newLinkedList();
  private Map<String, FailoverTimes> failoverNodes = Maps.newHashMap();
  private int priority;

  private FailoverTimes failoverTimesCheck(String role, int index) {
    String key = role + ":" + index;
    FailoverTimes failTimes = failoverNodes.get(key);
    if (failTimes == null) {
      failTimes = new FailoverTimes();
      failoverNodes.put(key, failTimes);
    }
    failTimes.failoverTimes++;
    LOG.info("node {}:{} fail times {}", role, index, failTimes);
    return failTimes;
  }

  private static class FailoverTimes {
    int failoverTimes;

    public int getFailTimes() {
      return failoverTimes;
    }

    @Override
    public String toString() {
      return "FailoverTimes [failoverTimes=" + failoverTimes + "]";
    }
  }

  public static class Request {
    public ContainerRequest request;
    public Container container;
    public String role;
    public int index;
    public FailoverTimes times;

    public Request(ContainerRequest request, String role, int index, FailoverTimes times) {
      this.request = request;
      this.role = role;
      this.index = index;
      this.times = times;
    }

    @Override
    public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("Request: [");
      sb.append("role: ").append(role).append(", ");
      sb.append("index: ").append(index).append(", ");
      sb.append("request: ").append(request).append(", ");
      sb.append("failtimes: ").append(times).append(", ");
      sb.append("]");
      return sb.toString();
    }
  }

  private void createFailoverRequest(Container container, int index, String role) {
    FailoverTimes times = failoverTimesCheck(role, index);

    Priority p = Records.newRecord(Priority.class);
    p.setPriority(priority);
    ContainerRequest containerAsk;
    if (role == Constants.PS_JOB_NAME) {
      containerAsk = new ContainerRequest(this.psContainerCapability, null, null, p);
    } else if (role == Constants.WORKER_JOB_NAME) {
      containerAsk = new ContainerRequest(this.workerContainerCapability, null, null, p);
    } else if (role == Constants.SCHEDULER_JOB_NAME) {
      containerAsk = new ContainerRequest(this.schedulerCapability, null, null, p);
    } else {
      containerAsk = new ContainerRequest(this.extendRoleContainerCapability.get(role), null, null, p);
    }
    failoverRequests.add(new Request(containerAsk, role, index, times));

    metaHandler.updateContainer(container.getId().toString(), Status.FAIL);
    rmClient.releaseAssignedContainer(container.getId());
  }

  private void requestFailoverNodes(Container container, int index, String jobName) throws Exception {
    createFailoverRequest(container, index, jobName);

    long startTime = System.currentTimeMillis();
    while (!failoverRequests.isEmpty()) {
      for (Request request : failoverRequests) {
        rmClient.addContainerRequest(request.request);
      }

      AllocateResponse response = rmClient.allocate(responseId);
      List<Container> listContainers = response.getAllocatedContainers();
      LOG.info("{} container has reallocated ", listContainers.size());
      if (listContainers.size() > 0) {
        outer: for (Container c : listContainers) {
          LOG.info("response container {}", c.toString());
          Iterator<Request> iter = failoverRequests.iterator();
          while (iter.hasNext()) {
            Request request = iter.next();
            if (matchesResource(request.request.getCapability(), c.getResource())) {
              request.container = c;
              LOG.info("response container {} matches request {}", c.toString(), request.toString());
              failoverLaunchRequests.add(request);
              iter.remove();
              rmClient.removeContainerRequest(request.request);
              continue outer;
            } else {
              LOG.info("response container {} not matches request {}", c.toString(), request.toString());
            }
          }
          rmClient.releaseAssignedContainer(c.getId());
          LOG.info("response container {} not matches and release", c.toString());
        }
      }

      for (Request request : failoverRequests) {
        rmClient.removeContainerRequest(request.request);
      }

      processResponse(response.getCompletedContainersStatuses(), nmClient);

      if (System.currentTimeMillis() - startTime > schedulerConf.max_failover_wait_secs * 1000) {
        throw new RuntimeException("failover wait resources timeout.");
      }

      Thread.sleep(2000);
    }
  }

  private boolean matchesResource(Resource request, Resource response) {
    if (response.getVirtualCores() == request.getVirtualCores() && response.getResourceValue("yarn.io/gpu") == request.getResourceValue("yarn.io/gpu")
        && response.getMemorySize() >= request.getMemorySize()) {
      return true;
    }
    return false;
  }

  private void launchForFailover() throws Exception {
    if (failoverLaunchRequests.isEmpty()) {
      return;
    }
    priority++;

    for (Request request : failoverLaunchRequests) {
      launchFailoverNode(request);
    }

    metaHandler.flush();
    failoverRequests.clear();
    failoverLaunchRequests.clear();
  }

  private void launchFailoverNode(Request request) throws Exception {
    if (request.role.equals(Constants.WORKER_JOB_NAME)) {
      workContainer.put(request.container.getId(), request.index);
      launchWorkerContainers[request.index] = request.container;
      launchContainer(nmClient, request.container, Constants.WORKER_JOB_NAME, request.index, null, request.times);
    } else if (request.role.equals(Constants.PS_JOB_NAME)) {
      psContainer.put(request.container.getId(), request.index);
      launchPsContainers[request.index] = request.container;
      launchContainer(nmClient, request.container, Constants.PS_JOB_NAME, request.index, null, request.times);
    } else if (request.role.equals(Constants.SCHEDULER_JOB_NAME)) {
      scheduler = request.container;
      launchContainer(nmClient, request.container, Constants.SCHEDULER_JOB_NAME, request.index, null, request.times);
    } else {
      extendRoleContainer.get(request.role).put(request.container.getId(), request.index);
      launchExtendRoleContainers.get(request.role)[request.index] = request.container;
      launchContainer(nmClient, request.container, request.role, request.index, null, request.times);
    }

    LOG.info("failover launch node {}:{} sucess!", request.role, request.index);
  }

  private ContainerRequest newSchedulerRequest(Priority p) {
    schedulerCapability.setMemorySize(4096);
    schedulerCapability.setVirtualCores(4);
    return new ContainerRequest(schedulerCapability, null, null, p);
  }

  private ContainerRequest newPsRequest(Priority p) {
    return new ContainerRequest(this.psContainerCapability, null, null, p);
  }

  private ContainerRequest newWorkerRequest(Priority p) {
    return new ContainerRequest(this.workerContainerCapability, null, null, p);
  }

  private ContainerRequest newExtendRoleRequest(String worker_type, Priority p) {
    return new ContainerRequest(this.extendRoleContainerCapability.get(worker_type), null, null, p);
  }

  private void requestSchedulerContainers(long startTime, int schedulerNumber) throws Exception {
    if (schedulerNumber <= 0) {
      return;
    }
    List<ContainerRequest> schContainerRequestList = new ArrayList<ContainerRequest>();
    LOG.info("request {} scheduler containers", schedulerNumber);
    this.addSchedulerRequest(schContainerRequestList, schedulerNumber);

    int launchSchContainerNumber = 0;
    List<Container> listSchContainersAllocated = new ArrayList<Container>();
    while (launchSchContainerNumber < schedulerNumber) {
      AllocateResponse response = rmClient.allocate(responseId);
      List<Container> listContainers = response.getAllocatedContainers();
      for (Container container : listContainers) {
        listSchContainersAllocated.add(container);
      }
      launchSchContainerNumber = listSchContainersAllocated.size();
      LOG.info("finish request {} scheduler containers", launchSchContainerNumber);
      if (launchSchContainerNumber >= schedulerNumber) {
        scheduler = listSchContainersAllocated.get(0);
        if (launchSchContainerNumber > schedulerNumber) {
          for (int i = 0; i > launchSchContainerNumber - schedulerNumber; i++) {
            ContainerId id = listSchContainersAllocated.get(launchSchContainerNumber - i - 1).getId();
            rmClient.releaseAssignedContainer(id);
            LOG.info("release scheduler container {}", id.toString());
          }
        }
      }

      if (System.currentTimeMillis() - startTime > schedulerConf.max_failover_wait_secs * 1000) {
        throw new RuntimeException("scheudler wait resources timeout.");
      }
      Thread.sleep(200);
    }
    LOG.info("finish request all scheduler containers");
    for (ContainerRequest containerRequest : schContainerRequestList) {
      rmClient.removeContainerRequest(containerRequest);
    }
  }

  private void requestWorkerContainers(long startTime, int workNumber) throws Exception {
    List<ContainerRequest> workerContainerRequestList = new ArrayList<ContainerRequest>();
    addWorkerRequest(workerContainerRequestList, workNumber);

    int launchWorkerContainerNumber = 0;
    List<Container> listWorkerContainersAllocated = new ArrayList<Container>();
    while (launchWorkerContainerNumber < workNumber) {
      AllocateResponse response = rmClient.allocate(responseId);
      List<Container> listContainers = response.getAllocatedContainers();
      for (Container container : listContainers) {
        listWorkerContainersAllocated.add(container);
      }
      launchWorkerContainerNumber = listWorkerContainersAllocated.size();
      LOG.info("finish request {} worker containers,total worker container is {}", launchWorkerContainerNumber,
          workNumber);
      if (launchWorkerContainerNumber >= workNumber) {
        launchWorkerContainers = new Container[workNumber];
        for (int i = 0; i < workNumber; i++) {
          launchWorkerContainers[i] = listWorkerContainersAllocated.get(i);
        }

        if (launchWorkerContainerNumber > workNumber) {
          for (int i = 0; i < launchWorkerContainerNumber - workNumber; i++) {
            ContainerId id = listWorkerContainersAllocated.get(launchWorkerContainerNumber - i - 1).getId();
            rmClient.releaseAssignedContainer(id);
            LOG.info("release worker container {}", id.toString());
          }
        }
      }
      if (System.currentTimeMillis() - startTime > schedulerConf.max_failover_wait_secs * 1000) {
        throw new RuntimeException("Worker wait resources timeout.");
      }
      Thread.sleep(200);
    }
    LOG.info("finish request all worker containers");
    for (ContainerRequest containerRequest : workerContainerRequestList) {
      rmClient.removeContainerRequest(containerRequest);
    }
  }

  private void requestExtendRoleContainers(long startTime, Map<String, Integer> extendRoleNum) throws Exception {
    Iterator<Map.Entry<String, Integer>> iter = extendRoleNum.entrySet().iterator();
    while (iter.hasNext()) {
      Map.Entry<String, Integer> entry = iter.next();
      String work_type = entry.getKey();
      int work_num = entry.getValue();
      List<ContainerRequest> workerContainerRequestList = new ArrayList<ContainerRequest>();
      addExtendRoleRequest(work_type, workerContainerRequestList, work_num);

      int launchWorkerContainerNumber = 0;
      List<Container> listWorkerContainersAllocated = new ArrayList<Container>();
      while (launchWorkerContainerNumber < work_num) {
        AllocateResponse response = rmClient.allocate(responseId);
        List<Container> listContainers = response.getAllocatedContainers();
        for (Container container : listContainers) {
          listWorkerContainersAllocated.add(container);
        }
        launchWorkerContainerNumber = listWorkerContainersAllocated.size();
        if (launchWorkerContainerNumber >= work_num) {
          launchExtendRoleContainers.put(work_type, new Container[work_num]);
          for (int i = 0; i < work_num; i++) {
            launchExtendRoleContainers.get(work_type)[i] = listWorkerContainersAllocated.get(i);
          }
          if (launchWorkerContainerNumber > work_num) {
            for (int i = 0; i < launchWorkerContainerNumber - work_num; i++) {
              ContainerId id = listWorkerContainersAllocated.get(launchWorkerContainerNumber - i - 1).getId();
              rmClient.releaseAssignedContainer(id);
              LOG.info("release {} container {}", work_type, id.toString());
            }
          }
        }
        if (System.currentTimeMillis() - startTime > schedulerConf.max_failover_wait_secs * 1000) {
          throw new RuntimeException(work_type + " wait resources timeout.");
        }
        Thread.sleep(200);
      }
      LOG.info("finish request all {} containers", work_type);
      for (ContainerRequest containerRequest : workerContainerRequestList) {
        rmClient.removeContainerRequest(containerRequest);
      }
      this.priority++;
    }
  }

  private void addWorkerRequest(List<ContainerRequest> workerContainerRequestList, int workNumber) {
    Priority p = Records.newRecord(Priority.class);
    p.setPriority(priority);
    for (int i = 0; i < workNumber; ++i) {
      ContainerRequest containerAsk = newWorkerRequest(p);
      LOG.info("Making resource request for worker container " + i);
      rmClient.addContainerRequest(containerAsk);
      workerContainerRequestList.add(containerAsk);
    }
  }

  private void addExtendRoleRequest(String worker_type, List<ContainerRequest> workerContainerRequestList,
      int workNumber) {
    Priority p = Records.newRecord(Priority.class);
    p.setPriority(priority);
    for (int i = 0; i < workNumber; ++i) {
      ContainerRequest containerAsk = newExtendRoleRequest(worker_type, p);
      LOG.info("Making resource request for {} container {}", worker_type, i);
      rmClient.addContainerRequest(containerAsk);
      workerContainerRequestList.add(containerAsk);
    }
  }

  private void addPsRequest(List<ContainerRequest> psContainerRequestList, int psNumber) {
    Priority p = Records.newRecord(Priority.class);
    p.setPriority(priority);
    for (int i = 0; i < psNumber; ++i) {
      ContainerRequest containerAsk = newPsRequest(p);
      LOG.info("Making resource request for ps container " + i);
      rmClient.addContainerRequest(containerAsk);
      psContainerRequestList.add(containerAsk);
    }
  }

  private void addSchedulerRequest(List<ContainerRequest> schContainerRequestList, int schNumber) {
    Priority p = Records.newRecord(Priority.class);
    p.setPriority(priority);
    for (int i = 0; i < schNumber; ++i) {
      ContainerRequest containerAsk = newSchedulerRequest(p);
      LOG.info("Making resource request for scheduler container " + i);
      rmClient.addContainerRequest(containerAsk);
      schContainerRequestList.add(containerAsk);
    }
  }

  private void requestPsContainers(long startTime, int psNumber) throws Exception {
    if (psNumber == 0) {
      launchPsContainers = new Container[0];
      return;
    }
    List<ContainerRequest> psContainerRequestList = new ArrayList<ContainerRequest>();
    LOG.info("request {} ps containers", psNumber);
    this.addPsRequest(psContainerRequestList, psNumber);

    int launchPsContainerNumber = 0;
    List<Container> listPsContainersAllocated = new ArrayList<Container>();
    while (launchPsContainerNumber < psNumber) {
      AllocateResponse response = rmClient.allocate(responseId);
      List<Container> listContainers = response.getAllocatedContainers();
      for (Container container : listContainers) {
        listPsContainersAllocated.add(container);
      }
      launchPsContainerNumber = listPsContainersAllocated.size();
      LOG.info("finish request {} ps containers", launchPsContainerNumber);
      if (launchPsContainerNumber >= psNumber) {
        launchPsContainers = new Container[psNumber];
        for (int i = 0; i < psNumber; i++) {
          launchPsContainers[i] = listPsContainersAllocated.get(i);
        }
        if (launchPsContainerNumber > psNumber) {
          for (int i = 0; i > launchPsContainerNumber - psNumber; i++) {
            ContainerId id = listPsContainersAllocated.get(launchPsContainerNumber - i - 1).getId();
            rmClient.releaseAssignedContainer(id);
            LOG.info("release ps container {}", id.toString());
          }
        }
      }

      if (System.currentTimeMillis() - startTime > schedulerConf.max_failover_wait_secs * 1000) {
        throw new RuntimeException("Ps wait resources timeout.");
      }
      Thread.sleep(200);
    }
    LOG.info("finish request all ps containers");
    for (ContainerRequest containerRequest : psContainerRequestList) {
      rmClient.removeContainerRequest(containerRequest);
    }
  }

  private void initResource() throws Exception {
    this.workNumber = this.schedulerConf.worker.instance_num;
    this.psNumber = this.schedulerConf.ps.instance_num;
    {
      long memory = schedulerConf.worker.memory_m;
      int cpuCores = schedulerConf.worker.cpu_cores;
      long gpuCores = schedulerConf.worker.gpu_cores;
      /*if (memory * cpuCores < 3096) {
        if (cpuCores > 0) {
          memory = 3096 / cpuCores + 1;
        } else {
          memory = 3096;
        }
        LOG.info("change work memory to {}", memory);
      }*/
      if (memory > maxResource.getMemorySize()) {
        memory = maxResource.getMemorySize();
      }
      if (cpuCores > maxResource.getVirtualCores()) {
        cpuCores = maxResource.getVirtualCores();
      }
      if (gpuCores > maxResource.getResourceValue("yarn.io/gpu")) {
        gpuCores = maxResource.getResourceValue("yarn.io/gpu");
      }
      LOG.info("worker instance gpu count {}", gpuCores);
      this.workerContainerCapability.setMemorySize(memory);
      this.workerContainerCapability.setVirtualCores(cpuCores);
      this.workerContainerCapability.setResourceValue("yarn.io/gpu", gpuCores);

      LOG.info("Worker container capability:[CPU: {}, GPU: {}, Memory: {} MB", cpuCores, gpuCores, memory);
    }
    if (this.schedulerConf.extend_role != null) {
      Iterator<Map.Entry<String, ExtendRoleResource>> iter = this.schedulerConf.extend_role.entrySet().iterator();
      while (iter.hasNext()) {
        Map.Entry<String, ExtendRoleResource> entry = iter.next();
        String key = entry.getKey();
        int val_num = entry.getValue().instance_num;
        this.extendRoleNum.put(key, val_num);
        long memory = entry.getValue().memory_m;
        int cpuCores = entry.getValue().cpu_cores;
        long gpuCores = entry.getValue().gpu_cores;
        if (memory * cpuCores < 3096) {
          if (cpuCores > 0) {
            memory = 3096 / cpuCores + 1;
          } else {
            memory = 3096;
          }
          LOG.info("change {} memory to {}", key, memory);
        }
        if (memory > maxResource.getMemorySize()) {
          memory = maxResource.getMemorySize();
        }
        if (cpuCores > maxResource.getVirtualCores()) {
          cpuCores = maxResource.getVirtualCores();
        }
        if (gpuCores > maxResource.getResourceValue("yarn.io/gpu")) {
          gpuCores = maxResource.getResourceValue("yarn.io/gpu");
        }
        Resource rr = Records.newRecord(Resource.class);
        rr.setMemorySize(memory);
        rr.setVirtualCores(cpuCores);
        rr.setResourceValue("yarn.io/gpu", gpuCores);
        this.extendRoleContainerCapability.put(key, rr);
        LOG.info("{} container capability:[CPU: {}, GPU: {}, Memory: {} MB", key, cpuCores, gpuCores, memory);
      }
    }
    {
      long memory = schedulerConf.ps.memory_m;
      int cpuCores = schedulerConf.ps.cpu_cores;
      long gpuCores = schedulerConf.ps.gpu_cores;
      if (memory * cpuCores < 3096) {
        if (cpuCores > 0) {
          memory = 3096 / cpuCores + 1;
        } else {
          memory = 3096;
        }
        LOG.info("change ps memory to {}", memory);
      }
      if (memory > maxResource.getMemorySize()) {
        memory = maxResource.getMemorySize();
      }
      if (cpuCores > maxResource.getVirtualCores()) {
        cpuCores = maxResource.getVirtualCores();
      }
      if (gpuCores > maxResource.getResourceValue("yarn.io/gpu")) {
        gpuCores = maxResource.getResourceValue("yarn.io/gpu");
      }
      this.psContainerCapability.setMemorySize(memory);
      this.psContainerCapability.setVirtualCores(cpuCores);
      this.psContainerCapability.setResourceValue("yarn.io/gpu", gpuCores);
      LOG.info("PS container capability:[CPU: {}, GPU: {}, Memory: {} MB", cpuCores, gpuCores, memory);
    }
    LOG.info("App:[{}] has [{}] ps job and [{}] worker job", this.appId, this.psNumber, this.workNumber);
    this.localResources = setupLocalResource(this.basePath);
    this.containerEnv = setupContainerEnv();
  }

  private Map<String, LocalResource> setupLocalResource(String basePath) throws IOException {
    Map<String, LocalResource> resourceMap = new HashMap<String, LocalResource>();
    FileSystem fs = FileSystem.get(yarnConf);
    FileStatus[] stats = fs.listStatus(new Path(basePath));
    for (FileStatus stat : stats) {
      if (stat.isFile()) {
        LOG.info(stat.getPath().toString());
        Path filePath = stat.getPath();
        LocalResource appContainerFile = Records.newRecord(LocalResource.class);
        FileStatus fileStat = FileSystem.get(yarnConf).getFileStatus(filePath);
        String fileName = filePath.getName();
        boolean isArchive = false;
        if (fileName.endsWith(".tar") || fileName.endsWith(".gz")) {
          isArchive = true;
        }
        appContainerFile.setResource(URL.fromPath(filePath));
        appContainerFile.setSize(fileStat.getLen());
        appContainerFile.setTimestamp(fileStat.getModificationTime());
        if (isArchive) {
          appContainerFile.setType(LocalResourceType.ARCHIVE);
        } else {
          appContainerFile.setType(LocalResourceType.FILE);
        }
        appContainerFile.setVisibility(LocalResourceVisibility.PUBLIC);
        resourceMap.put(fileName, appContainerFile);
      }
    }
    fs.close();
    return resourceMap;
  }

  private Map<String, String> setupContainerEnv() {
    String javaPathSeparator = System.getProperty("path.separator");
    Map<String, String> appContainerEnv = new HashMap<String, String>();
    Apps.addToEnvironment(appContainerEnv, ApplicationConstants.Environment.CLASSPATH.name(),
        ApplicationConstants.Environment.PWD.$() + File.separator + "*", javaPathSeparator);

    for (String c : yarnConf.getStrings(YarnConfiguration.YARN_APPLICATION_CLASSPATH,
        YarnConfiguration.DEFAULT_YARN_APPLICATION_CLASSPATH)) {
      Apps.addToEnvironment(appContainerEnv, ApplicationConstants.Environment.CLASSPATH.name(), c.trim(),
          javaPathSeparator);
    }

    LOG.info("JAVA CLASS_PATH is " + System.getProperty("java.class.path"));
    return appContainerEnv;
  }

  public void dealWithExit(Status status) {
    try {
      switch (status) {
      case KILLED:
        this.rmClient.unregisterApplicationMaster(FinalApplicationStatus.KILLED, "", "");
        break;
      case FINISHED:
        this.rmClient.unregisterApplicationMaster(FinalApplicationStatus.SUCCEEDED, "", "");
        break;
      default:
        this.rmClient.unregisterApplicationMaster(FinalApplicationStatus.FAILED, "", "");
        break;
      }
    } catch (Exception e) {
      LOG.error("deal with exit error!", e.getMessage());
    }

    try {
      this.metaHandler.udpateJobStatus(status);
      this.metaHandler.flush();
      this.metaHandler.close();
    } catch (Exception e) {
      LOG.error("deal with exit error!", e.getMessage());
    }

    try {
      this.close();
    } catch (Exception e) {
      LOG.error("deal with exit error!", e.getMessage());
    }

    if (nmClient != null) {
      try {
        nmClient.close();
      } catch (IOException e) {
        LOG.error("deal with exit error!", e.getMessage());
      }
    }
    if (rmClient != null) {
      try {
        rmClient.close();
      } catch (IOException e) {
        LOG.error("deal with exit error!", e.getMessage());
      }
    }
  }

  protected void removeZkRootPath() throws Exception {
    String appZKRootPath = Paths.get("/psplus", this.appId).toString();
    if (zkCli.checkExists().forPath(appZKRootPath) != null) {
      zkCli.delete().deletingChildrenIfNeeded().forPath(appZKRootPath);
      LOG.info("App Master remove zookeeper app path: [{}]", appZKRootPath);
    }
  }

  protected void closeZkCli() throws Exception {
    if (zkCli != null) {
      try {
        zkCli.close();
      } catch (Exception e) {
        LOG.error("close zk client error!", e);
      }
    }
  }

  protected void close() throws Exception {
    removeZkRootPath();
    closeZkCli();
  }

  private String generateApplicationMasterWorkDir() {
    String workDir = Utils.getCurDir();
    String host = null;
    try {
      host = InetAddress.getLocalHost().toString();
    } catch (UnknownHostException e) {
      LOG.warn("Get AM host failed.");
    }
    return String.format("%s:%s", host, workDir);
  }

  private AMRMClient<ContainerRequest> rmClient;
  private NMClient nmClient;
  private Resource maxResource;

  private void createRMClient() throws Exception {
    this.rmClient = AMRMClient.createAMRMClient();
    rmClient.init(yarnConf);
    rmClient.start();
    LOG.info("ResourceManager client started.");
    RegisterApplicationMasterResponse response = rmClient.registerApplicationMaster("", 0,
        generateApplicationMasterWorkDir());
    maxResource = response.getMaximumResourceCapability();
 
    if (maxResource.getResourceValue("yarn.io/gpu") == 0) {
        maxResource.setResourceValue("yarn.io/gpu", 4);
    }

    LOG.info("ApplicationMaster max memory [{}] max cpu cores [{}] max gpu cores [{}]", maxResource.getMemorySize(),
        maxResource.getVirtualCores(), maxResource.getResourceValue("yarn.io/gpu"));
    LOG.info("Register ApplicationMaster success.");
  }

  private void createNMClient() throws Exception {
    nmClient = NMClient.createNMClient();
    nmClient.init(yarnConf);
    nmClient.start();
    LOG.info("NodeManager client started.");
  }

  protected void init() throws Exception {}

  protected void preRun() throws Exception {}
}
