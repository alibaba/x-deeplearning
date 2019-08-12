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

#include <atomic>
#include <string>
#include <thread>
#include <mutex>
#include <unistd.h>
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
  void AddCallback(std::function<void(Status)> func);
  cudaStream_t GetInternal() { return internal_; }
  void Lock() { mu_.lock(); }
  void Unlock() { mu_.unlock(); }
  static void RunOrAbort(cudaError_t status, const char* msg);
  bool QueryAndRunCallbacks();

 private:
  cudaStream_t internal_;
  std::mutex mu_;

  std::vector<std::function<void(Status)>> callbacks_;
  size_t callbacks_num_ = 0;
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

class CudaStreams {
 public:
  static CudaStreams* GetInstance() {
   static CudaStreams instance;
   return &instance;
  }
  CudaStream* GetCudaStream() {
    size_t index = index_.fetch_add(1) % kStreamsCapacity;
    CudaStream* cuda_stream = cuda_streams_[index];
    //cudaStreamSynchronize(cuda_stream->GetInternal());
    return cuda_stream;
  }
  virtual ~CudaStreams() {
    alive_ = false;
    cb_thread_.join();
    for (int i = 0; i < kStreamsCapacity; ++i) {
      delete cuda_streams_[i];
    }
  }

 private:
  CudaStreams() : alive_(true), index_(0) {
    cuda_streams_.reserve(kStreamsCapacity);
    for (int i = 0; i < kStreamsCapacity; ++i) {
      cuda_streams_.push_back(new CudaStream());
    }
    cb_thread_ = std::thread(&CudaStreams::CallbackMain, this);
  }
  void CallbackMain() {
    while (alive_) {
      bool ret = false;
      size_t index = index_ % kStreamsCapacity;
      size_t i = index;
      while (alive_) {
        ret |= cuda_streams_[i]->QueryAndRunCallbacks();
        if (++i >= kStreamsCapacity) i -= kStreamsCapacity;
        if (i == index) break;
      }
      if (!ret) {
        usleep(100);
      }
    }
  }
  bool alive_;
  std::thread cb_thread_;
  std::atomic<size_t> index_;
  std::vector<CudaStream*> cuda_streams_;
  static constexpr int kStreamsCapacity = 64;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_GPU_STREAM_H_

