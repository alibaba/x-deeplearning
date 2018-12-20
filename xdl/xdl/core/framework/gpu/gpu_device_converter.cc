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
#include "xdl/core/framework/gpu/gpu_stream.h"
#include "xdl/core/framework/gpu/gpu_device.h"
#include "xdl/core/framework/cpu_device.h"

namespace xdl {

class GpuCpuConverter : public DeviceConverter {
 public:
  GpuCpuConverter() : stream_(nullptr) {}
  void Convert(Device* src_device, Device* dst_device,
               Tensor src, Tensor* dst,
               ThreadPool* tp, std::function<void(Status)> cb) override {
    GpuDevice* src_gpu = dynamic_cast<GpuDevice*>(src_device);
    CpuDevice* dst_cpu = dynamic_cast<CpuDevice*>(dst_device);
    XDL_CHECK_COND_ASYNC(src_gpu != nullptr,
        Status::Internal("Src Device Should Be GPU"), cb);
    XDL_CHECK_COND_ASYNC(dst_cpu != nullptr,
        Status::Internal("Dst Device Should Be CPU"), cb);
    if (stream_ == nullptr) {
      std::unique_lock<std::mutex> lock(mu_);
      if (stream_ == nullptr) {
        stream_ = CudaStreamManager::Instance()->GetCudaStream(kStream);
      }
    }
    *dst = Tensor(dst_cpu, src.Shape(), src.Type());
    stream_->Lock();
    Buffer* src_buffer = src.GetBuffer();
    Buffer* dst_buffer = dst->GetBuffer();
    CudaStream::RunOrAbort(
        cudaMemcpyAsync(dst_buffer->begin(), src_buffer->begin(),
                        src_buffer->size(), cudaMemcpyDeviceToHost,
                        stream_->GetInternal()),
        "cuda Memcpy device to host Error");
    stream_->AddCallback(tp, cb);
    stream_->Unlock();
  }
 private:
  static constexpr int kStream = -2;
  std::mutex mu_;
  CudaStream* stream_;
};

class CpuGpuConverter : public DeviceConverter {
 public:
  CpuGpuConverter() : stream_(nullptr) {}
  void Convert(Device* src_device, Device* dst_device,
               Tensor src, Tensor* dst,
               ThreadPool* tp, std::function<void(Status)> cb) override {
    CpuDevice* src_cpu = dynamic_cast<CpuDevice*>(src_device);
    GpuDevice* dst_gpu = dynamic_cast<GpuDevice*>(dst_device);
    XDL_CHECK_COND_ASYNC(src_cpu != nullptr,
        Status::Internal("Src Device Should Be CPU"), cb);
    XDL_CHECK_COND_ASYNC(dst_gpu != nullptr,
        Status::Internal("Dst Device Should Be GPU"), cb);
    if (stream_ == nullptr) {
      std::unique_lock<std::mutex> lock(mu_);
      if (stream_ == nullptr) {
        stream_ = CudaStreamManager::Instance()->GetCudaStream(kStream);
      }
    }
    *dst = Tensor(dst_gpu, src.Shape(), src.Type());
    stream_->Lock();
    Buffer* src_buffer = src.GetBuffer();
    Buffer* dst_buffer = dst->GetBuffer();
    CudaStream::RunOrAbort(
        cudaMemcpyAsync(dst_buffer->begin(), src_buffer->begin(),
                        src_buffer->size(), cudaMemcpyHostToDevice,
                        stream_->GetInternal()),
        "cuda Memcpy host to device Error");
    stream_->AddCallback(tp, cb);
    stream_->Unlock();
  }
 private:
  static constexpr int kStream = -3;
  std::mutex mu_;
  CudaStream* stream_;
};

class GpuGpuConverter : public DeviceConverter {
  void Convert(Device* src_device, Device* dst_device,
               Tensor src, Tensor* dst,
               ThreadPool* tp, std::function<void(Status)> cb) override {
    //TODO: Add 2 Device
    *dst = src;
    cb(Status::Ok());
  }
};

XDL_DEVICE_CONVERTER_REGISTER(GpuCpuConverter, GPU, CPU);
XDL_DEVICE_CONVERTER_REGISTER(CpuGpuConverter, CPU, GPU);
XDL_DEVICE_CONVERTER_REGISTER(GpuGpuConverter, GPU, GPU);

}  // namespace xdl

