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

#include "xdl/core/framework/device_registry.h"

namespace xdl {

Status DeviceRegistry::GetDevice(const DeviceDef& device_def,
                                 Device** device) {
  std::unique_lock<std::mutex> lock(cache_mu_);
  auto iter = cache_.find(device_def);
  if (iter != cache_.end()) {
    *device = iter->second.get();
    return Status::Ok();
  }
  auto factory_iter = factories_.find(device_def.device_name);
  XDL_CHECK_COND(factory_iter != factories_.end(),
                 Status::ArgumentError("device def not found "
                                       + device_def.device_name));
  Factory& factory = factory_iter->second;
  Device* result;
  XDL_CHECK_STATUS(factory(device_def, &result));
  cache_[device_def].reset(result);
  *device = result;
  return Status::Ok();
}

void DeviceRegistry::RegisterDevice(const std::string& name,
                                    DeviceRegistry::Factory factory) {
  factories_[name] = factory;
}

size_t DeviceRegistry::DeviceDefHash::operator()(const DeviceDef& def) const {
  static const size_t P1 = 98765431, P2 = 10240319;
  std::hash<std::string> str_hasher;
  size_t s = str_hasher(def.device_name) * P1;
  for (auto&& item : def.attr) {
    s += str_hasher(item.first) * P2 + str_hasher(item.second);
  }
  return s;
}

bool DeviceRegistry::DeviceDefEqual::operator()(const DeviceDef& lhs,
                                                const DeviceDef& rhs) const {
  if (lhs.device_name != rhs.device_name) {
    return false;
  }
  if (lhs.attr.size() != rhs.attr.size()) {
    return false;
  }
  for (auto&& item : lhs.attr) {
    auto iter = rhs.attr.find(item.first);
    if (iter == rhs.attr.end()) {
      return false;
    }
    if (iter->second != item.second) {
      return false;
    }
  }
  return true;
}

}  // namespace xdl

