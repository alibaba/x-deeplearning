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

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <array>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <cstring>

#include "hdfs_define.h"
#include "hdfs_launcher.h"

using namespace std;

namespace xdl {

HDFSLauncher::HDFSLauncher() { 
  config_path_ = std::string(XDL_HADOOP_PATH) + "etc/hadoop/core-site.xml";
  hdfs_config_path_ = std::string(XDL_HADOOP_PATH) + "etc/hadoop/hdfs-site.xml";
  Start();
}

HDFSLauncher::~HDFSLauncher() { 
  Stop();
}

int HDFSLauncher::GetAvailablePort() {
    struct sockaddr_in addr;
    addr.sin_port = htons(0);  // have system pick up a random port available for me
    addr.sin_family = AF_INET;  // IPV4
    addr.sin_addr.s_addr = htonl(INADDR_ANY);  // set our addr to any interface

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (0 != bind(sock, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
        return -1;
    }

    socklen_t addr_len = sizeof(struct sockaddr_in);
    if (0 != getsockname(sock, (struct sockaddr*)&addr, &addr_len)) {
        return -1;
    }

    int ret_port = ntohs(addr.sin_port);
    close(sock);
    return ret_port;
}

void HDFSLauncher::GetAvailablePorts(int num, std::vector<int>* ports) {
  std::set<int> uniq_ports;
  while (uniq_ports.size() < num) {
    int port = GetAvailablePort();
    if (port == -1) {
	continue;
    }
    
    if (uniq_ports.find(port) == uniq_ports.end()) {
	uniq_ports.insert(port);
    }
  }

  ports->insert(ports->end(), uniq_ports.begin(), uniq_ports.end());
}

std::string HDFSLauncher::Exec(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
  if (!pipe) throw std::runtime_error("popen() failed!");
  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
      result += buffer.data();
  }
  return result;
}

void HDFSLauncher::Start() {
  port_ = GetAvailablePort();
  UpdateCoreSiteConfig(port_);
  std::vector<int> ports;
  GetAvailablePorts(5, &ports);
  UpdateHdfsSiteConfig(ports);
  stringstream ss;
  ss << "cd " << XDL_HADOOP_PATH << "; sh run.sh " << \
    std::to_string(port_) << "; sleep 26";
  string out = Exec(ss.str().c_str());
  cout << out << std::endl;
}

void HDFSLauncher::Stop() {
  stringstream ss;
  ss << "cd " << XDL_HADOOP_PATH << "; sh stop.sh";
  string out = Exec(ss.str().c_str());
  cout << out << std::endl;
  RestoreCoreSiteConfig();
  RestoreHdfsSiteConfig();
}

}

