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

#include "xdl/core/framework/device_converter.h"

namespace xdl {

std::string DeviceConverterRegistry::UniqId(
    const std::string& src, const std::string& dst) {
  return src + "->" + dst;
}

DeviceConverter* DeviceConverterRegistry::Get(
    const std::string& src, const std::string& dst) {
  auto iter = map_.find(UniqId(src, dst));
  if (iter == map_.end()) {
    return nullptr;
  }
  return iter->second.get();
}

void DeviceConverterRegistry::Register(
    DeviceConverter* converter,
    const std::string& src,
    const std::string& dst) {
  map_[UniqId(src, dst)].reset(converter);
}

}  // namespace xdl

