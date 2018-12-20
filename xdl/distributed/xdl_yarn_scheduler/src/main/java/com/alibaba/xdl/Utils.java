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
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.alibaba.fastjson.JSON;

import jersey.repackaged.com.google.common.collect.Lists;

public class Utils {

  private static Logger LOG = LoggerFactory.getLogger(Utils.class);

  public static final FsPermission permTemp = new FsPermission("777");
  public static final String HDFS_USER_ROOT = "/user";
  public static final String HDFS_XDL_WORK_DIR = ".xdl";
  public static final String HDFS_PYTHON_DIR = "python";

  public static String genAppBasePath(Configuration conf, String appId, String user) throws IOException {
    String basePath = Paths.get(HDFS_USER_ROOT, user, HDFS_XDL_WORK_DIR, appId).toString();
    if (basePath.startsWith("/")) {
      basePath = basePath.substring(1);
    }
    String defaultFs = conf.get("fs.defaultFS");
    basePath = defaultFs + basePath + "/";
    FileSystem fs = FileSystem.get(conf);
    Path appStagePath = new Path(basePath);
    if (!fs.exists(appStagePath)) {
      fs.mkdirs(appStagePath, permTemp);
      fs.setPermission(appStagePath, permTemp);
      LOG.info("Path:[{}] not exists, create success.", appStagePath);
    }
    return basePath;
  }

  public static String getCurUser() {
    return System.getProperty("user.name");
  }

  public static String getCurDir() {
    return System.getProperty("user.dir");
  }

  public static SchedulerConf parseSchedulerConf(String config) {
    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new FileReader(config));
      String line = null;
      StringBuilder json = new StringBuilder();
      while ((line = reader.readLine()) != null) {
        json.append(line).append("\n");
      }
      return JSON.parseObject(json.toString(), SchedulerConf.class);
    } catch (IOException e) {
      throw new RuntimeException("reader config error!", e);
    } finally {
      if (reader != null) {
        try {
          reader.close();
        } catch (Exception e) {
        }
      }
    }
  }

  public static JobConf parseJobConf(String config) {
    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new FileReader(config));
      String line = null;
      StringBuilder json = new StringBuilder();
      while ((line = reader.readLine()) != null) {
        json.append(line).append("\n");
      }

      @SuppressWarnings("unchecked")
      Map<String, Object> jobConf = (Map<String, Object>) JSON.parseObject(json.toString(), HashMap.class);

      return new JobConf(jobConf);
    } catch (IOException e) {
      throw new RuntimeException("reader config error!", e);
    } finally {
      if (reader != null) {
        try {
          reader.close();
        } catch (Exception e) {
        }
      }
    }
  }

  public static void runCmd(String cmd) {
    String[] cmdArr = cmd.split(" ");
    Runtime run = Runtime.getRuntime();
    try {
      Process p = run.exec(cmdArr);
      p.waitFor();
      if (p.exitValue() == 0) {
        LOG.info("Run cmd [" + cmd + "] success.");
        return;
      } else {
        BufferedReader read = new BufferedReader(new InputStreamReader(p.getErrorStream()));
        String result = read.readLine();
        throw new RuntimeException("Run cmd [" + cmd + "] failed, " + result);
      }
    } catch (Exception e) {
      e.printStackTrace();
      throw new RuntimeException(e.getMessage());
    }
  }

  public static String extractFileName(String filePath) {
    return new Path(filePath).getName();
  }

  public static void logFinalStatus(String appId, String status) {
    LOG.info("================================FINAL STATUS==================================");
    LOG.info("  {} : {} ", appId, status);
    LOG.info("================================FINAL STATUS==================================");
  }

  public static void restartLog(String message) {
    LOG.info("==================================================================================================");
    LOG.info("===" + message);
    LOG.info("==================================================================================================");
    System.out
        .println("==================================================================================================");
    System.out.println("===" + message);
    System.out
        .println("==================================================================================================");
  }

  public static ArrayList<String> splitString(String value, String seperator) {
    if (value.isEmpty()) {
      return Lists.newArrayList();
    }
    return new ArrayList<String>(Arrays.asList(value.split(seperator)));
  }

  public static void removeHdfsDirs(Configuration conf, String dir) throws IOException {
    FileSystem dfs = null;
    try {
      String userFs = dir.substring(0, dir.indexOf("/", dir.indexOf("//") + 2));
      FileSystem.setDefaultUri(conf, userFs);
      dfs = FileSystem.get(conf);

      Path path = new Path(dir);
      if (dfs.exists(path)) {
        dfs.delete(path, true);
        LOG.info("Delete the hdfs dir:{} success.", dir);
      }
    } catch (IOException e) {
      e.printStackTrace();
      throw new RuntimeException("delete hdfs dirs error!", e);
    } finally {
      try {
        dfs.close();
      } catch (Exception e) {
      }
    }
  }

  public static void createHdfsDirs(FileSystem dfs, Configuration conf, String... paths) throws Exception {
    for (String p : paths) {
      Path dir = new Path(p);
      if (!dfs.exists(dir)) {
        dfs.mkdirs(dir, permTemp);
      }
    }
  }

  public static void writeHdfsFile(Configuration conf, String dir, String fileName, String content) throws Exception {
    FileSystem dfs = null;
    PrintWriter out = null;
    try {
      dfs = FileSystem.get(conf);

      createHdfsDirs(dfs, conf, dir);

      Path aucFile = null;
      if (dir.endsWith("/")) {
        aucFile = new Path(dir + fileName);
      } else {
        aucFile = new Path(dir + "/" + fileName);
      }

      out = new PrintWriter(dfs.create(aucFile, true));
      out.print(content);
      out.flush();

    } finally {
      try {
        dfs.close();
      } catch (Exception e) {
      }

      try {
        out.close();
      } catch (Exception e) {
      }
    }
  }

  public static boolean existsHdfsFile(Configuration conf, String dir) throws Exception {
    if (dir == null) {
      return false;
    }
    FileSystem dfs = null;
    try {
      String userFs = dir.substring(0, dir.indexOf("/", dir.indexOf("//") + 2));
      FileSystem.setDefaultUri(conf, userFs);
      dfs = FileSystem.get(conf);

      Path aucFile = new Path(dir);

      if (dfs.exists(aucFile)) {
        return true;
      } else {
        return false;
      }
    } finally {
      try {
        dfs.close();
      } catch (Exception e) {
      }
    }
  }
}
