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

#include "xdl/core/framework/gpu/gpu_stream.h"

#include <iostream>

namespace xdl {

namespace {
class CudaCallbackClosure {
 public:
  CudaCallbackClosure(ThreadPool* tp, std::function<void(Status)> cb)
    : tp_(tp), cb_(cb) {}
  void SetStatus(Status st) {
    st_ = st;
  }
  void Run() {
    tp_->Schedule([this]{
      cb_(st_);
      delete this;
    });
  }
 private:
  ThreadPool* tp_;
  std::function<void(Status)> cb_;
  Status st_;
};

void CUDART_CB CudaCallback(
    cudaStream_t stream, cudaError_t status, void* user_data) {
  CudaCallbackClosure* closure = static_cast<CudaCallbackClosure*>(user_data);
  if (status != cudaSuccess) {
    closure->SetStatus(Status::Internal("Cuda Error Code "
                                        + std::to_string(status)));
  }
  closure->Run();
}

}  // namespace

CudaStream::CudaStream() {
  RunOrAbort(cudaStreamCreate(&internal_), "Cuda Stream Create Error");
}

void CudaStream::AddCallback(
    //ThreadPool* tp,
    std::function<void(Status)> func) {
#if 0
  RunOrAbort(cudaStreamAddCallback(
      internal_, CudaCallback, new CudaCallbackClosure(tp, func), 0),
      "Cuda Add Callback Erro");
#else
  std::lock_guard<std::mutex> lock(mu_);
  callbacks_.push_back(func);
  ++ callbacks_num_;
#endif
}

bool CudaStream::QueryAndRunCallbacks() {
  if (callbacks_num_ > 0) {
    std::lock_guard<std::mutex> lock(mu_);
    if (cudaStreamQuery(internal_) == cudaSuccess) {
      for (auto& callback : callbacks_) callback(Status::Ok());
      callbacks_.clear();
      callbacks_num_ = 0;
      return true;
    }
  }
  return false;
}

void CudaStream::RunOrAbort(cudaError_t status, const char* msg) {
  if (status != cudaSuccess && status != cudaErrorCudartUnloading) {
    std::cerr << msg << " Error Code: " << status << std::endl;
    abort();
  }
}

}  // namespace xdl
