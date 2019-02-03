/*
 * \file context.cc
 * \brief The context for operator run on CPU/GPU
 */
#include "blaze/common/context.h"

#include "blaze/common/log.h"

namespace blaze {

CopyFunction g_copy_function[kMaxDevNum][kMaxDevNum];

#ifdef USE_CUDA
thread_local ThreadLocalCUDAObjects CUDAContext::cuda_objects_;

CUDAContext::CUDAContext(const int gpu_id) : gpu_id_(gpu_id) { Activate(); }

CUDAContext::CUDAContext(const DeviceOption& device_option) {
  BLAZE_CONDITION_THROW(device_option.device_type() == kCUDA,
                        "device_option device_type is not kCUDA");
  if (device_option.has_device_id()) {
    gpu_id_ = device_option.device_id();
  } else {
    gpu_id_ = 0;
  }
}

// device memory copy functions
void GPUCopy(void* dst, const void* src, size_t dst_offset, size_t src_offset, size_t size, void* stream) {
  CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char*>(dst) + dst_offset,
                             reinterpret_cast<const char*>(src) + src_offset,
                             size,
                             cudaMemcpyDefault,
                             reinterpret_cast<cudaStream_t>(stream)));
}

REGISTER_COPY_FUNCTION(kCUDA, kCUDA, GPUCopy)
REGISTER_COPY_FUNCTION(kCUDA, kCPU, GPUCopy)
REGISTER_COPY_FUNCTION(kCPU, kCUDA, GPUCopy)

#endif

void CPUCopy(void* dst, const void* src, size_t dst_offset, size_t src_offset, size_t size, void* stream) {
  memcpy(reinterpret_cast<char*>(dst) + dst_offset,
         reinterpret_cast<const char*>(src) + src_offset,
         size);
}

REGISTER_COPY_FUNCTION(kCPU, kCPU, CPUCopy)

}  // namespace blaze

