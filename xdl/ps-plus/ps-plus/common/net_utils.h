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

#ifndef PS_COMMON_NET_UTILS_H_
#define PS_COMMON_NET_UTILS_H_

#include <string>
#include <vector>
#include <stdlib.h>

namespace ps {

class NetUtils {
 public:
  static bool GetHostName(std::string& hostName);
  static bool GetIP(std::vector<std::string>& ips);
  static bool GetDefaultIP(std::string& ip);
  static std::string GetLocalIP(const std::string& interface);
  static int GetAvailablePort();
  static int GetAvailableCpuNum();
  static std::string GetEnv(const std::string& name) {
    char* s = getenv(name.c_str());
    if (s == nullptr) {
      return "";
    }

    return std::string(s);
  }
};

} //ps

#endif  // PS_COMMON_NET_UTILS_H
