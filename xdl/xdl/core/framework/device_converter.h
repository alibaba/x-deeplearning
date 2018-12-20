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

#ifndef XDL_CORE_FRAMEWORK_DEVICE_CONVERTER_H_
#define XDL_CORE_FRAMEWORK_DEVICE_CONVERTER_H_

#include "xdl/core/lib/singleton.h"
#include "xdl/core/framework/device.h"
#include "xdl/core/framework/tensor.h"

namespace xdl {

class DeviceConverter {
 public:
  virtual ~DeviceConverter() {}
  virtual void Convert(Device* src_device, Device* dst_device,
                       Tensor src, Tensor* dst,
                       ThreadPool* tp, std::function<void(Status)> cb) = 0;
};

class DeviceConverterRegistry : public Singleton<DeviceConverterRegistry> {
 public:
  DeviceConverter* Get(const std::string& src, const std::string& dst);
  void Register(DeviceConverter* converter,
                const std::string& src,
                const std::string& dst);
  std::string UniqId(const std::string& src, const std::string& dst);
 private:
  std::unordered_map<std::string, std::unique_ptr<DeviceConverter>> map_;
};

class DeviceConverterRegisterHelper {
 public:
  DeviceConverterRegisterHelper(
      DeviceConverter* converter,
      const std::string& src,
      const std::string& dst) {
    DeviceConverterRegistry::Instance()->Register(converter, src, dst);
  }
};

}  // namespace xdl

#define XDL_DEVICE_CONVERTER_REGISTER(type, src, dst)                    \
  XDL_DEVICE_CONVERTER_REGISTER_UNIQ_HELPER(__COUNTER__, type, src, dst)
#define XDL_DEVICE_CONVERTER_REGISTER_UNIQ_HELPER(ctr, type, src, dst)   \
  XDL_DEVICE_CONVERTER_REGISTER_UNIQ(ctr, type, src, dst)
#define XDL_DEVICE_CONVERTER_REGISTER_UNIQ(ctr, type, src, dst)          \
  static ::xdl::DeviceConverterRegisterHelper __attribute__((unused))    \
  __register_device_converter__##ctr(new type, #src, #dst);

#endif  // XDL_CORE_FRAMEWORK_DEVICE_CONVERTER_H_
