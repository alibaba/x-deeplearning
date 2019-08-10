/* Copyright 2018 Alibaba Group. All Rights Reserved.

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

#ifndef XDL_BACKEND_DEVICE_SINGLETON_
#define XDL_BACKEND_DEVICE_SINGLETON_

#include "xdl/core/framework/cpu_device.h"
#ifdef USE_GPU
#include "xdl/core/framework/gpu/gpu_device.h"
#endif

namespace xdl {

class DeviceSingleton {
 public:
  static CpuDevice* CpuInstance() {
    static CpuDevice cpu_device;
    return &cpu_device;
  }

#ifdef USE_GPU
  static GpuDevice* GpuInstance() {
    static GpuDevice gpu_device(1);
    return &gpu_device;
  }
#endif

 protected:
  DeviceSingleton() {}
};

} // namespace xdl

#endif // XDL_BACKEND_DEVICE_SINGLETON_
