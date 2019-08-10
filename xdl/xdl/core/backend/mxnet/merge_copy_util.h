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

#ifndef XDL_BACKEND_MXNET_MERGE_COPY_UTIL_H_
#define XDL_BACKEND_MXNET_MERGE_COPY_UTIL_H_

#include <omp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>
#include "xdl/core/utils/logging.h"

#include "xdl/core/lib/status.h"
#include "xdl/core/lib/common_defines.h"
#include "xdl/core/framework/tensor.h"
#include "xdl/core/framework/gpu/gpu_stream.h"
#include "xdl/core/utils/time_utils.h"

namespace xdl {

class MergeCopyUtil {
 public:
  MergeCopyUtil() 
    : forward_size_(0) 
    , backward_size_(0)
    , forward_device_buf_(nullptr)
    , backward_device_buf_(nullptr) {}

  inline void RegisterForwardMemInfo(const std::string& name,
                              void* device_addr,
                              size_t size);
  inline void RegisterBackwardMemInfo(const std::string& name,
                               void* device_addr,
                               size_t size);
  inline void Initialize() {
    // forward
    forward_buf_.resize(forward_size_);
    cudaError_t e = cudaMalloc(&forward_device_buf_, forward_size_);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      size_t avail, total;
      cudaMemGetInfo(&avail, &total);
      LOG(ERROR) << "cudaMalloc failed, size=" << forward_size_
                 << " avail=" << avail
                 << " total=" << total
                 << " err=" << e;
    }

    CudaStream::RunOrAbort(e, "Cuda Memory Allocate Error");
    // backward
    backward_buf_.resize(backward_size_);
    e = cudaMalloc(&backward_device_buf_, backward_size_);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      size_t avail, total;
      cudaMemGetInfo(&avail, &total);
      LOG(ERROR) << "cudaMalloc failed, size=" << backward_size_
                 << " avail=" << avail
                 << " total=" << total
                 << " err=" << e;
    }

    CudaStream::RunOrAbort(e, "Cuda Memory Allocate Error");

    // cuda stream
    CudaStream::RunOrAbort(cudaStreamCreate(&stream_), "Cuda Stream Create Error");    
  }

  inline Status CopyHost2Device(const std::vector<std::pair<std::string, Tensor> >& tensors);
  inline Status CopyDevice2Host(const std::vector<std::string>& output_order,
                                std::vector<Tensor>* tensors);

 private:
  struct DeviceMemInfo {
    DeviceMemInfo() : device_addr_(nullptr), size_(0), offset_(0) {}
    DeviceMemInfo(void* device_addr,
                  size_t size,
                  size_t offset) 
      : device_addr_(device_addr)
      , size_(size)
      , offset_(offset) {}
    void* device_addr_;
    size_t size_;
    size_t offset_;
  };

  std::unordered_map<std::string, DeviceMemInfo> forward_mem_info_;
  std::unordered_map<std::string, DeviceMemInfo> backward_mem_info_;
  size_t forward_size_;
  size_t backward_size_;
  std::vector<char> forward_buf_;
  std::vector<char> backward_buf_;  
  void* forward_device_buf_;
  void* backward_device_buf_;
  cudaStream_t stream_;  
};

void MergeCopyUtil::RegisterForwardMemInfo(
    const std::string& name,
    void* device_addr,
    size_t size) {
  forward_mem_info_.insert(std::make_pair(name, DeviceMemInfo(device_addr, size, forward_size_)));
  forward_size_ += size;
}

void MergeCopyUtil::RegisterBackwardMemInfo(
    const std::string& name,
    void* device_addr,
    size_t size) {
  backward_mem_info_.insert(std::make_pair(name, DeviceMemInfo(device_addr, size, backward_size_)));
  backward_size_ += size;
}

Status MergeCopyUtil::CopyHost2Device(const std::vector<std::pair<std::string, Tensor> >& tensors) {
  size_t beg = TimeUtils::NowMicros();    
  //#pragma omp parallel for num_threads(2)
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto it = forward_mem_info_.find(tensors[i].first);
    if (it == forward_mem_info_.end()) {
//      return Status::ArgumentError("no input:" + tensors[i].first + " registered!");
      continue;
    }

    void* dst = (void*)forward_buf_.data() + it->second.offset_;
    memcpy(dst, tensors[i].second.Raw<char>(), it->second.size_);
  }

  size_t beg1 = TimeUtils::NowMicros();
  CUDA_CHECK(cudaMemcpy(forward_device_buf_,
                        forward_buf_.data(),
                        forward_size_,
                        cudaMemcpyHostToDevice));

  size_t beg2 = TimeUtils::NowMicros();
  for (auto& item: forward_mem_info_) {
    CUDA_CHECK(cudaMemcpyAsync(item.second.device_addr_,
                               forward_device_buf_ + item.second.offset_,
                               item.second.size_,
                               cudaMemcpyDeviceToDevice,
                               stream_));
  }
  
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  return Status::Ok();
}

Status MergeCopyUtil::CopyDevice2Host(const std::vector<std::string>& output_order,
                                      std::vector<Tensor>* outputs) {
  size_t beg = TimeUtils::NowMicros();    
  for (auto& item: backward_mem_info_) {
    CUDA_CHECK(cudaMemcpyAsync(backward_device_buf_ + item.second.offset_,
                               item.second.device_addr_,
                               item.second.size_,
                               cudaMemcpyDeviceToDevice,
                               stream_));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream_));
  size_t beg1 = TimeUtils::NowMicros();    
  CUDA_CHECK(cudaMemcpy(backward_buf_.data(),
                        backward_device_buf_,
                        backward_size_,
                        cudaMemcpyDeviceToHost));

  size_t beg2 = TimeUtils::NowMicros();    
  //#pragma omp parallel for num_threads(2)
  for (size_t i = 0; i < output_order.size(); ++i) {
    auto it = backward_mem_info_.find(output_order[i]);
    if (it == backward_mem_info_.end()) {
      // return Status::ArgumentError("no input:" + output_order[i] + " registered!");
      continue;
    }

    memcpy(outputs->at(i).Raw<void>(), backward_buf_.data() + it->second.offset_, it->second.size_);
  }

  return Status::Ok();
}


} // namespace xdl

#endif // XDL_BACKEND_MXNET_MERGE_COPY_UTIL_H_
