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

#include "ps-plus/common/reliable_kv.h"

namespace ps {

Status ReliableKV::ReadAny(const std::string& addr, std::string* val, int retry) {
  std::size_t pos = addr.find(':');
  if (pos == std::string::npos) {
    return Status::NotFound("ReliableKV: Address Error: " + addr);
  }
  std::string protocol = addr.substr(0, pos);
  ReliableKV* engine = GetPlugin<ReliableKV>(protocol);
  if (engine == nullptr) {
    return Status::NotFound("ReliableKV: Protocol Error: " + protocol);
  }
  return engine->Read(addr, val, retry);
}

Status ReliableKV::WriteAny(const std::string& addr, const std::string& val, int retry) {
  std::size_t pos = addr.find(':');
  if (pos == std::string::npos) {
    return Status::NotFound("ReliableKV: Address Error: " + addr);
  }
  std::string protocol = addr.substr(0, pos);
  ReliableKV* engine = GetPlugin<ReliableKV>(protocol);
  if (engine == nullptr) {
    return Status::NotFound("ReliableKV: Protocol Error: " + protocol);
  }
  return engine->Write(addr, val, retry);
}

};

