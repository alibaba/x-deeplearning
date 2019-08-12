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

#ifndef HDFS_LAUNCHER_H_
#define HDFS_LAUNCHER_H_

#include <string>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <vector>
#include <set>

namespace xdl {
  
class HDFSLauncher {
public:
  void Start();
  void Stop();
  static HDFSLauncher* Instance() {
    static HDFSLauncher launch;
    return &launch;
  }

  int GetPort() {
    return port_;
  }

private:
  HDFSLauncher();
  ~HDFSLauncher();

  std::string ReadLocalFile(const std::string &filePath) {
    std::ifstream in(filePath.c_str());
    std::stringstream ss;
    std::string line;
    if (!in) {
      return std::string("");
    }
    while (std::getline(in, line)) {
      ss << line;
    }
    in.close();
    return ss.str();
  }

  bool WriteLocalFile(const std::string &filePath, 
		      const std::string &content) {
    std::ofstream file(filePath.c_str());
    if (!file) {
      return false;
    }
    file.write(content.c_str(), content.length());
    file.flush();
    file.close();
    return true;
  }

  int GetAvailablePort();
  void GetAvailablePorts(int num, std::vector<int>* ports);

  void UpdateCoreSiteConfig(int port) {
    raw_config_ = ReadLocalFile(config_path_);
    std::string new_config = raw_config_;
    size_t pos = new_config.find("${port}");
    if (pos != std::string::npos) {
      new_config.replace(pos, 7, std::to_string(port));
    }

    WriteLocalFile(config_path_, new_config);
  }

  void RestoreCoreSiteConfig() {
    WriteLocalFile(config_path_, raw_config_);
  }

  void UpdateHdfsSiteConfig(const std::vector<int>& port_list) {
    raw_hdfs_config_ = ReadLocalFile(hdfs_config_path_);
    std::string hdfs_config = raw_hdfs_config_;
    for (size_t i = 1; i <= port_list.size(); ++i) {
      std::string placeholder = "${port" + std::to_string(i) + "}";
      std::string value = "0.0.0.0:" + std::to_string(port_list[i]);
      size_t pos = hdfs_config.find(placeholder);
      if (pos != std::string::npos) {
	hdfs_config.replace(pos, placeholder.size(), value);
      }
    }

    WriteLocalFile(hdfs_config_path_, hdfs_config);
  }

  void RestoreHdfsSiteConfig() {
    WriteLocalFile(hdfs_config_path_, raw_hdfs_config_);
  }
  
  static std::string Exec(const char *cmd);
  int port_;
  std::string config_path_;
  std::string raw_config_;
  std::string hdfs_config_path_;
  std::string raw_hdfs_config_;
};

}

#endif
