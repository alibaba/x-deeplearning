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

#include "ps-plus/common/net_utils.h"

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
#include "ps-plus/common/logging.h"

namespace ps {

bool NetUtils::GetHostName(std::string& hostName) {
  char buf[128];
  if (0 == gethostname(buf, sizeof(buf))) {
    hostName = buf;
    return true;
  }
  return false;
}

bool NetUtils::GetIP(std::vector<std::string>& ips) {
  std::string hostName;
  if (!GetHostName(hostName)) {
    return false;
  }
  struct hostent* hent;
  hent = gethostbyname(hostName.c_str());
  for (uint32_t i = 0; hent->h_addr_list[i]; i++) {
    std::string ip = inet_ntoa(*(struct in_addr*)(hent->h_addr_list[i]));
    ips.push_back(ip);
  }
  return true;
}

bool NetUtils::GetDefaultIP(std::string& ip) {
  std::vector<std::string> ips;
  if (!GetIP(ips)) return false;
  if (ips.empty()) return false;
  ip = ips[0];
  return true;
}

std::string NetUtils::GetLocalIP(const std::string& interface) {
  struct ifaddrs * ifAddrStruct = NULL;
  struct ifaddrs * ifa = NULL;
  void * tmpAddrPtr = NULL;
  getifaddrs(&ifAddrStruct);
  std::string ip;
  for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == NULL) continue;
    if (ifa->ifa_addr->sa_family == AF_INET) {
      // is a valid IP4 Address
      tmpAddrPtr = &(reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr))->sin_addr;
      char addressBuffer[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
      if (!interface.empty()) {
        if (strncmp(ifa->ifa_name,
                    interface.c_str(),
                    interface.size()) == 0) {
          ip = addressBuffer;
          break;
        }
      } else {
        ip = addressBuffer;
        break;
      }
    }
  }

  if (ifAddrStruct != NULL) { 
    freeifaddrs(ifAddrStruct);
  }

  return ip;
}

int NetUtils::GetAvailablePort() {
  struct sockaddr_in addr;
  addr.sin_port = htons(0);  // have system pick up a random port available for me
  addr.sin_family = AF_INET;  // IPV4
  addr.sin_addr.s_addr = htonl(INADDR_ANY);  // set our addr to any interface

  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (0 != bind(sock, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
    LOG(ERROR) << "bind failed";
    return -1;
  }

  socklen_t addr_len = sizeof(struct sockaddr_in);
  if (0 != getsockname(sock, (struct sockaddr*)&addr, &addr_len)) {
    LOG(ERROR) << "getsockname failed!";
    return -1;
  }

  int ret_port = ntohs(addr.sin_port);
  close(sock);
  return ret_port;
}

int NetUtils::GetAvailableCpuNum() {
  char buf[16] = {0};
  int num;
  FILE* fp = popen("cat /proc/cpuinfo |grep processor|wc -l", "r");
  if (fp) {
    fread(buf, 1, sizeof(buf) - 1, fp);
    pclose(fp);
  }   
  num = atoi(buf);
  if (num <= 0){ 
    num = 1;
  }   

  return num;
}

} //ps

