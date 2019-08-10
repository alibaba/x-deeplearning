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
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <netinet/in.h>

#include "zookeeper_launcher.h"
#include "zookeeper_define.h"

using namespace std;

namespace xdl {

ZookeeperLauncher::ZookeeperLauncher() {
  config_path_ = std::string(XDL_ZOOKEEPER_PATH) + "conf/zoo.cfg";
  Start();
}

ZookeeperLauncher::~ZookeeperLauncher() {
  Stop();
}

int ZookeeperLauncher::GetAvailablePort() {
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

std::string ZookeeperLauncher::Exec(const char* cmd) {
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

void ZookeeperLauncher::Start(void) {
  port_ = GetAvailablePort();
  UpdateCoreSiteConfig(port_);
  stringstream ss;
  ss << "cd " << XDL_ZOOKEEPER_PATH << "; bin/zkServer.sh start; sleep 26";
  string out = Exec(ss.str().c_str());
  cout << out << std::endl;
}

void ZookeeperLauncher::Stop(void) {
  std::cout << "Stop zookeeeper!\n\n\n";
  stringstream ss;
  ss << "cd " << XDL_ZOOKEEPER_PATH << "; bin/zkServer.sh stop; rm -rf zookeeper.out";
  string out = Exec(ss.str().c_str());
  cout << out << std::endl;
  out = Exec("rm -rf /tmp/xdl_zookeeper");
  cout << out << std::endl;
  RestoreCoreSiteConfig();
}

}
