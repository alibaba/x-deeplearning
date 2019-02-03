/*
 * \file context.h 
 * \brief The context for operator run on GPU
 */
#pragma once

#include <mutex>
#include <vector>

#include "blaze/common/allocator.h"
#include "blaze/common/common_defines.h"
#include "blaze/common/event.h"
#include "blaze/common/exception.h"
#include "blaze/proto/blaze.pb.h"

namespace blaze {

class CPUContext {
 public:
  CPUContext() { }
  explicit CPUContext(const DeviceOption& device_option) {
    CHECK(device_option.device_type() == kCPU,
          "device_option's device_type not equal");
  }

  inline void SwitchToDevice(int stream_id = 0) { }
  inline void Activate() { }
  inline void WaitEvent(const Event& ev) {
    ev.Wait(kCPU, this);
  }
  inline void Record(Event* ev, const char* err_msg = nullptr) const {
    CHECK(ev != nullptr, "Event must not be nullptr");
    ev->Record(kCPU, this, err_msg);
  }
  inline void FinishDeviceComputation() { }
 
  int stream_id() const { return 0; }
  int device_id() const { return 0; }
  static int device_type() { return kCPU; }

  // CPU Stream is useless.
  static bool StreamIsFree(const DeviceOption&, int) { return true; }
  bool HasAsyncPart() const { return false; }
};

#ifdef USE_CUDA
// In Blaze, each thread has its's own non-default cuda streams.
class ThreadLocalCUDAObjects {
  friend class CUDAContext;
 
 private:
  ThreadLocalCUDAObjects() {
    for (int i = 0; i < kMaxGPUNum; ++i) {
      cuda_streams_[i] = std::vector<cudaStream_t>();
      cublas_handles_[i] = std::vector<cublasHandle_t>();
      cudnn_handles_[i] = std::vector<cudnnHandle_t>();
    }
  }
  cudaStream_t GetStream(int gpu, int stream_id) {
    std::vector<cudaStream_t>& gpu_streams = cuda_streams_[gpu];
    if (gpu_streams.size() <= stream_id) {
      gpu_streams.resize(stream_id + 1, 0);
    }
    if (!gpu_streams[stream_id]) {
      CUDADeviceGuard guard(gpu);
      CUDA_CHECK(cudaStreamCreateWithFlags(&gpu_streams[stream_id], cudaStreamNonBlocking));
    }
    return gpu_streams[stream_id];
  }
  cublasHandle_t GetHandle(int gpu, int stream_id) {
    std::vector<cublasHandle_t>& gpu_handles = cublas_handles_[gpu];
    if (gpu_handles.size() <= stream_id) {
      gpu_handles.resize(stream_id + 1, nullptr);
    }
    if (gpu_handles[stream_id] == nullptr) {
      CUDADeviceGuard guard(gpu);
      CUBLAS_CHECK(cublasCreate(&gpu_handles[stream_id]));
      CUBLAS_CHECK(cublasSetStream(gpu_handles[stream_id], GetStream(gpu, stream_id)));
    }
    return gpu_handles[stream_id];
  }
  cudnnHandle_t GetDnnHandle(int gpu, int stream_id) {
    std::vector<cudnnHandle_t>& gpu_handles = cudnn_handles_[gpu];
    if (gpu_handles.size() <= stream_id) {
      gpu_handles.resize(stream_id + 1, nullptr);
    }
    if (gpu_handles[stream_id] == nullptr) {
      CUDADeviceGuard guard(gpu);
      CUDNN_CHECK(cudnnCreate(&gpu_handles[stream_id]));
      CUDNN_CHECK(cudnnSetStream(gpu_handles[stream_id], GetStream(gpu, stream_id)));
    }
    return gpu_handles[stream_id];
  }
  ~ThreadLocalCUDAObjects() noexcept {
    for (int i = 0; i < kMaxGPUNum; ++i) {
      for (auto& handle : cublas_handles_[i]) {
        if (handle) {
          cublasDestroy(handle);
        }
      }
      for (auto& handle : cudnn_handles_[i]) {
        if (handle) {
          cudnnDestroy(handle);
        }
      }
      for (auto& stream : cuda_streams_[i]) {
        if (stream) {
          cudaStreamDestroy(stream);
        }
      }
    }
  }

  static const int kMaxGPUNum = 4;
  std::vector<cudaStream_t> cuda_streams_[kMaxGPUNum];
  std::vector<cublasHandle_t> cublas_handles_[kMaxGPUNum];
  std::vector<cudnnHandle_t> cudnn_handles_[kMaxGPUNum];
};

// The maximum stream per GPU
#define kMaxStreamNum 4

class CUDAContext {
 public:
  explicit CUDAContext(const int gpu_id = 0);
  explicit CUDAContext(const DeviceOption& device_option);

  inline void SwitchToDevice(int stream_id = 0) {
    stream_id_ = stream_id;
    this->Activate();
  }
  inline void Activate() {
    CUDA_CHECK(cudaSetDevice(gpu_id_));
  }
  inline cudaStream_t cuda_stream() {
    return cuda_objects_.GetStream(gpu_id_, stream_id_);
  }
  inline cudaStream_t cuda_stream() const {
    return cuda_objects_.GetStream(gpu_id_, stream_id_);
  }
  inline cublasHandle_t cublas_handle() {
    return cuda_objects_.GetHandle(gpu_id_, stream_id_);
  }
  inline cudnnHandle_t cudnn_handle() {
    return cuda_objects_.GetDnnHandle(gpu_id_, stream_id_);
  }
  static cudaStream_t cuda_stream(int gpu_id, int stream_id) {
    return cuda_objects_.GetStream(gpu_id, stream_id);
  }
  static cublasHandle_t cublas_handle(int gpu_id, int stream_id) {
    return cuda_objects_.GetHandle(gpu_id, stream_id);
  }
  static cudnnHandle_t cudnn_handle(int gpu_id, int stream_id) {
    return cuda_objects_.GetDnnHandle(gpu_id, stream_id);
  }
  inline void WaitEvent(const Event& ev) {
    ev.Wait(kCUDA, this);
  }
  inline void Record(Event* ev, const char* err_msg = nullptr) const {
    CHECK(ev != nullptr, "ev is nullptr");
    ev->Record(kCUDA, this, err_msg);
  }
  inline void FinishDeviceComputation() {
    CUDA_CHECK(cudaStreamSynchronize(cuda_objects_.GetStream(gpu_id_, stream_id_)));
  }

  int stream_id() const { return stream_id_; }
  int device_id() const { return gpu_id_; }
  static int device_type() { return kCUDA; }

  static bool StreamIsFree(const DeviceOption& device_option, int stream_id) {
    auto stream = cuda_objects_.GetStream(device_option.device_id(), stream_id);
    return cudaStreamQuery(stream) == cudaSuccess;
  }

  bool HasAsyncPart() const { return true; }

 protected:
  int stream_id_ = 0;
  int gpu_id_ = 0;
  static thread_local ThreadLocalCUDAObjects cuda_objects_;
};
#endif

template <typename Context>
inline DeviceOption GetDeviceOption(int device_id = 0) {
  DeviceOption device_option;
  device_option.set_device_type(Context::device_type());
  device_option.set_device_id(device_id);
  return device_option;
}

#ifdef USE_CUDA
#define CONTEXT_SWITCH(context_type, DType, ...)                 \
    switch (context_type) {                                      \
      case DeviceType::kCPU:                                     \
        {                                                        \
          typedef blaze::CPUContext DType;                       \
          {__VA_ARGS__}                                          \
        }                                                        \
        break;                                                   \
      case DeviceType::kCUDA:                                    \
        {                                                        \
          typedef blaze::CUDAContext DType;                      \
          {__VA_ARGS__}                                          \
        }                                                        \
        break;                                                   \
      default:                                                   \
        {                                                        \
          BLAZE_THROW("Unsupport context_type ", context_type);  \
        }                                                        \
    }
#else
#define CONTEXT_SWITCH(context_type, DType, ...)                 \
    switch (context_type) {                                      \
      case DeviceType::kCPU:                                     \
        {                                                        \
          typedef blaze::CPUContext DType;                       \
          {__VA_ARGS__}                                          \
        }                                                        \
        break;                                                   \
      default:                                                   \
        {                                                        \
          BLAZE_THROW("Unsupport context_type ", context_type);  \
        }                                                        \
    }
#endif

// Copy function
typedef std::function<void(void* dst,
                           const void* src,
                           size_t dst_offset,
                           size_t src_offset,
                           size_t size,
                           void*)> CopyFunction;
#define kMaxDevNum 8
extern CopyFunction g_copy_function[kMaxDevNum][kMaxDevNum];

// The copy function register
struct CopyFunctionRegister {
  CopyFunctionRegister(int dst_type, int src_type, CopyFunction cf) {
    g_copy_function[dst_type][src_type] = cf;
  }
};

#ifndef REGISTER_COPY_FUNCTION
#define REGISTER_COPY_FUNCTION(dst_type, src_type, func) \
    CopyFunctionRegister ANONYMOUS_VARIABLE(cp)(dst_type, src_type, func);
#endif

}  // namespace blaze

