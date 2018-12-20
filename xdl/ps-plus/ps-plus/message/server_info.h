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

#ifndef PS_COMMON_SERVER_INFO_H_
#define PS_COMMON_SERVER_INFO_H_

#include <cstdint>
#include <sstream>
#include <string>

#include "version.h"

namespace ps {

using ServerId = uint64_t;
using ServerType = uint64_t;

class ServerInfo {
 public:
  ServerInfo(ServerType server_type,
             ServerId id, 
             Version version, 
             const std::string& ip, 
             uint16_t port)
    : server_type_(server_type)
    , id_(id)
    , version_(version)
    , ip_(ip)
    , port_(port) {}
  ServerInfo() = default;
  
  ServerType GetServerType() const noexcept { return server_type_; }
  ServerId GetId() const noexcept { return id_; }
  Version GetVersion() const noexcept { return version_; }
  const std::string& GetIp() const noexcept { return ip_; }
  uint32_t GetPort() const noexcept { return port_; }

  std::string Address() const noexcept {
    return ip_ + ":" + std::to_string(port_);
  }

  std::string ToString() const {
    std::ostringstream os;
    os << server_type_ << "-" << id_ << " (" << ip_ << ":" << port_ << "@" << version_ << ")";
    return os.str();
  }

  bool operator==(const ServerInfo& other) const {
    return (id_ == other.id_
            && version_ == other.version_
            && ip_ == other.ip_
            && port_ == other.port_);
  }

  bool operator!=(const ServerInfo& other) const { return !(*this == other); }

  ServerType server_type_;
  ServerId id_;
  Version version_;
  std::string ip_;
  uint16_t port_;
};

} // namespace ps

#endif // PS_COMMON_SERVER_INFO_H_
