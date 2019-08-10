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

#include <time.h>
#include <thread>
#include <unistd.h>

#include "xdl/core/framework/device_converter.h"
#include "xdl/core/framework/gpu/gpu_stream.h"
#include "xdl/core/framework/gpu/gpu_device.h"
#include "xdl/core/framework/cpu_device.h"
#include "xdl/core/lib/blocking_queue.h"

namespace xdl {

static double GetTime() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

class GpuTransferManager {
 public:
  struct TransferMsg {
    Buffer* src_buffer;
    Buffer* dst_buffer;
    std::function<void(Status)> cb;
    TransferMsg(Buffer* src_buffer = nullptr,Buffer* dst_buffer = nullptr,
                std::function<void(Status)> cb = nullptr)
    : src_buffer(src_buffer), dst_buffer(dst_buffer), cb(cb) {}
  };

  GpuTransferManager(CudaStream* stream, enum cudaMemcpyKind kind)
  : stream_(stream), kind_(kind), alive_(true) {
    queue_ = new BlockingQueue<TransferMsg>(kQueueCapacity);
    cb_queue_ = new BlockingQueue<TransferMsg>(kQueueCapacity);
    thread_ = std::thread(&GpuTransferManager::TransferMain, this);
    cb_thread_ = std::thread(&GpuTransferManager::CallbackMain, this);
  }
  virtual ~GpuTransferManager() {
    alive_ = false;
    queue_->Clear();
    cb_queue_->Clear();
    thread_.join();
    cb_thread_.join();
    delete queue_;
    delete cb_queue_;
  }

  void Enqueue(Buffer* src_buffer, Buffer* dst_buffer,
               std::function<void(Status)> cb) {
    while (!queue_->TryEnqueue(TransferMsg(src_buffer, dst_buffer, cb), 1) && alive_) {
      printf("Warning: transfer queue %d is full!\n", kind_);
    }
  }

  void TransferMain() {
    TransferMsg transfer_msg;
    std::vector<TransferMsg> cb_msgs;
    cb_msgs.reserve(kQueueCapacity);
    while (alive_) {
      double btime = GetTime();
      while (queue_->TryDequeue(&transfer_msg, 0)) {
        CudaStream::RunOrAbort(
            cudaMemcpyAsync(transfer_msg.dst_buffer->begin(), transfer_msg.src_buffer->begin(),
                            transfer_msg.src_buffer->size(), kind_,
                            stream_->GetInternal()),
            "cudaMemcpyAsync error");
        total_size_ += transfer_msg.src_buffer->size();
        ++round_;
        cb_msgs.push_back(transfer_msg);
      }
      size_t cb_msgs_size = cb_msgs.size();
      if (cb_msgs_size > 0) {
        CudaStream::RunOrAbort(
            cudaStreamSynchronize(stream_->GetInternal()),
            "cudaStreamSynchronize error");
        total_time_ += GetTime() - btime;
        if (round_ % 10000000 < cb_msgs_size) {
          printf("Transfer %d (1:CPU=>GPU,2:GPU=>CPU) speed = %g MB/s, %g round/s\n",
                 kind_, total_size_ / 1024 / 1024 / total_time_, round_ / total_time_);
        }
        for (size_t i = 0; i < cb_msgs_size; ++i) {
          while (!cb_queue_->TryEnqueue(cb_msgs[i], 1) && alive_) {
            printf("Warning: callback queue %d is full!\n", kind_);
          }
        }
        cb_msgs.clear();
      } else {
        usleep(1000);
      }
    }
  }

  void CallbackMain() {
    TransferMsg transfer_msg;
    while (alive_) {
      if (cb_queue_->TryDequeue(&transfer_msg, 1)) {
        transfer_msg.cb(Status());
      }
    }
  }

 private:
  CudaStream* stream_;
  enum cudaMemcpyKind kind_;
  bool alive_;
  BlockingQueue<TransferMsg>* queue_;
  BlockingQueue<TransferMsg>* cb_queue_;
  std::thread thread_;
  std::thread cb_thread_;
  static constexpr int kQueueCapacity = 65536;

  double total_time_ = 0.;
  size_t total_size_ = 0;
  size_t round_ = 0;
};

class GpuCpuConverter : public DeviceConverter {
 public:
  GpuCpuConverter() : stream_(nullptr), transfer_manager_(nullptr) {}
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
    if (transfer_manager_ == nullptr) {
      std::unique_lock<std::mutex> lock(mu_);
      if (transfer_manager_ == nullptr) {
        transfer_manager_ = new GpuTransferManager(stream_, cudaMemcpyDeviceToHost);
      }
    }
    *dst = Tensor(dst_cpu, src.Shape(), src.Type());
    //stream_->Lock();
    Buffer* src_buffer = src.GetBuffer();
    if (src_buffer->size() > 0) {
      Buffer* dst_buffer = dst->GetBuffer();
#if 0
      CudaStream::RunOrAbort(
          cudaMemcpyAsync(dst_buffer->begin(), src_buffer->begin(),
                          src_buffer->size(), cudaMemcpyDeviceToHost,
                          stream_->GetInternal()),
          "cuda Memcpy device to host Error");
      stream_->AddCallback(tp, cb);
#else
      transfer_manager_->Enqueue(src_buffer, dst_buffer, cb);
#endif
    } else {
      cb(Status());
    }
    //stream_->Unlock();
  }
 private:
  static constexpr int kStream = -2;
  std::mutex mu_;
  CudaStream* stream_;
  GpuTransferManager* transfer_manager_;
};

class CpuGpuConverter : public DeviceConverter {
 public:
  CpuGpuConverter() : stream_(nullptr), transfer_manager_(nullptr) {}
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
    if (transfer_manager_ == nullptr) {
      std::unique_lock<std::mutex> lock(mu_);
      if (transfer_manager_ == nullptr) {
        transfer_manager_ = new GpuTransferManager(stream_, cudaMemcpyHostToDevice);
      }
    }
    *dst = Tensor(dst_gpu, src.Shape(), src.Type());
    //stream_->Lock();
    Buffer* src_buffer = src.GetBuffer();
    if (src_buffer->size() > 0) {
      Buffer* dst_buffer = dst->GetBuffer();
#if 0
      CudaStream::RunOrAbort(
          cudaMemcpyAsync(dst_buffer->begin(), src_buffer->begin(),
                          src_buffer->size(), cudaMemcpyHostToDevice,
                          stream_->GetInternal()),
          "cuda Memcpy host to device Error");
      stream_->AddCallback(tp, cb);
#else
      transfer_manager_->Enqueue(src_buffer, dst_buffer, cb);
#endif
    } else {
      cb(Status());
    }
    //stream_->Unlock();
  }
 private:
  static constexpr int kStream = -3;
  std::mutex mu_;
  CudaStream* stream_;
  GpuTransferManager* transfer_manager_;
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

