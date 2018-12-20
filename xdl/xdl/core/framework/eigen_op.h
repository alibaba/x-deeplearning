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

#ifndef XDL_CORE_FRAMEWORK_EIGEN_OP_H_
#define XDL_CORE_FRAMEWORK_EIGEN_OP_H_

#include "xdl/core/framework/cpu_device.h"
#include "xdl/core/framework/gpu/gpu_device.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace xdl {
template <typename Device, typename Functor>
class EigenOp;

class ThreadPoolDeviceManager : public Singleton<ThreadPoolDeviceManager> {
 public:
  ThreadPoolDeviceManager()
    : thread_pool_(std::thread::hardware_concurrency()),
      internal_(&thread_pool_, std::thread::hardware_concurrency()) {}

  Eigen::ThreadPoolDevice* Get() {
    return &internal_;
  }
 private:
  Eigen::ThreadPool thread_pool_;
  Eigen::ThreadPoolDevice internal_;
};

template <typename Functor>
class EigenOp<CpuDevice, Functor> : public OpKernel {
  using EigenDevice = typename Eigen::ThreadPoolDevice;
  Status Compute(OpKernelContext* ctx) override {
    EigenDevice* device = ThreadPoolDeviceManager::Instance()->Get();
    XDL_CHECK_STATUS(Functor()(device, ctx));
    return Status::Ok();
  }
};

template <typename Functor>
class EigenOp<GpuDevice, Functor> : public GpuOpKernel {
  using EigenDevice = typename Eigen::GpuDevice;
  Status LaunchKernel(OpKernelContext* ctx, CudaStream* stream) override {
    cudaStream_t internal_stream = stream->GetInternal();
    Eigen::CudaStreamDevice stream_device(&internal_stream);
    Eigen::GpuDevice device(&stream_device);
    XDL_CHECK_STATUS(Functor()(&device, ctx));
    return Status::Ok();
  }
};

}  // namespace xdl

#endif

