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

#ifndef XDL_CORE_FRAMEWORK_GPU_STREAM_H_
#define XDL_CORE_FRAMEWORK_GPU_STREAM_H_

#include <string>
#include <mutex>
#include <unordered_map>
#include "cuda_runtime.h"
#include "xdl/core/utils/logging.h"

#include "xdl/core/lib/status.h"
#include "xdl/core/lib/singleton.h"
#include "xdl/core/lib/thread_pool.h"

namespace xdl {

class CudaStream {
 public:
  CudaStream();
  void AddCallback(ThreadPool* tp, std::function<void(Status)> func);
  cudaStream_t GetInternal() { return internal_; }
  void Lock() { mu_.lock(); }
  void Unlock() { mu_.unlock(); }
  static void RunOrAbort(cudaError_t status, const char* msg);
 private:
  cudaStream_t internal_;
  std::mutex mu_;
};

class CudaStreamManager : public Singleton<CudaStreamManager> {
 public:
  CudaStream* GetCudaStream(int id) {
    std::unique_lock<std::mutex> lock(mu_);
    auto iter = streams_.find(id);
    if (iter != streams_.end()) {
      return iter->second.get();
    }
    CudaStream* stream = new CudaStream;
    streams_[id].reset(stream);
    return stream;
  }
 private:
  std::mutex mu_;
  std::unordered_map<int, std::unique_ptr<CudaStream>> streams_;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_GPU_STREAM_H_

