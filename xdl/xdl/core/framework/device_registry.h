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

#ifndef XDL_CORE_FRAMEWORK_DEVICE_REGISTRY_H_
#define XDL_CORE_FRAMEWORK_DEVICE_REGISTRY_H_

#include <string>
#include <unordered_map>
#include <functional>
#include <mutex>

#include "xdl/core/lib/singleton.h"
#include "xdl/core/framework/device.h"

namespace xdl {

class DeviceRegistry : public Singleton<DeviceRegistry> {
 public:
  using Factory = std::function<Status(const DeviceDef& def, Device** device)>;
  Status GetDevice(const DeviceDef& device_def, Device** device);
  void RegisterDevice(const std::string& name, Factory factory);
 private:
  struct DeviceDefHash {
    size_t operator()(const DeviceDef& def) const;
  };
  struct DeviceDefEqual {
    bool operator()(const DeviceDef& lhs, const DeviceDef& rhs) const;
  };

  std::mutex cache_mu_;
  std::unordered_map<std::string, Factory> factories_;
  std::unordered_map<DeviceDef, std::unique_ptr<Device>,
                     DeviceDefHash, DeviceDefEqual> cache_;
};

class DeviceRegistryHelper {
 public:
  DeviceRegistryHelper(const std::string& name,
                       DeviceRegistry::Factory factory) {
    DeviceRegistry::Get()->RegisterDevice(name, factory);
  }
};

}  // namespace xdl

#define XDL_DEVICE_FACTORY(name, factory)                     \
  XDL_DEVICE_FACTORY_UNIQ_HELPER(__COUNTER__, name, factory)
#define XDL_DEVICE_FACTORY_UNIQ_HELPER(ctr, name, factory)    \
  XDL_DEVICE_FACTORY_UNIQ(ctr, name, factory)
#define XDL_DEVICE_FACTORY_UNIQ(ctr, name, factory)           \
  static ::xdl::DeviceRegistryHelper __attribute__((unused))  \
  __register_device__##ctr(#name, factory);

#endif  // XDL_CORE_FRAMEWORK_DEVICE_REGISTRY_H_

