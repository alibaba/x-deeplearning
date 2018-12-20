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

#ifndef PS_COMMON_CLUSTER_INFO_H_
#define PS_COMMON_CLUSTER_INFO_H_

#include <map>

#include "server_info.h"

namespace ps {

class ClusterInfo {
 public:
  const std::vector<ServerInfo>& GetServers() const noexcept {
    return servers_;
  }
  void AddServer(const ServerInfo& server) {
    servers_.push_back(server);
  }

  std::vector<ServerInfo> servers_;
  std::vector<size_t> server_size_;
};

} // namespace ps

#endif // PS_COMMON_CLUSTER_INFO_H_
