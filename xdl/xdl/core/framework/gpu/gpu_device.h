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

#ifndef XDL_CORE_FRAMEWORK_GPU_DEVICE_H_
#define XDL_CORE_FRAMEWORK_GPU_DEVICE_H_

#include <string>

#include "xdl/core/framework/device.h"
#include "xdl/core/framework/device_registry.h"
#include "xdl/core/framework/gpu/gpu_stream.h"
#include "xdl/core/framework/op_kernel.h"

namespace xdl {

class GpuAllocator : public Allocator {
 public:
  void* Allocate(size_t size) override;
  void Deallocate(void* buf) override;
};

class GpuDevice : public Device {
 public:
  std::string DeviceType() override;
  static Status CreateDevice(const DeviceDef& def, Device** device);
  CudaStream* Stream() { return stream_; }
  GpuDevice(int id);
 private:
  CudaStream* stream_;
};

class GpuOpKernel : public OpKernelBase {
 public:
  void Launch(OpKernelContext* ctx) override;
  virtual Status LaunchKernel(OpKernelContext* ctx, CudaStream* stream) = 0;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_GPU_DEVICE_H_

