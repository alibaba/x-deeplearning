/*
 * \file gpu_kernel_launcher.h
 * \desc gpu kernel launcher   
 */
#pragma once

#include "blaze/common/common_defines.h"
#include "blaze/common/cuda_helpers.h"

namespace blaze {

template<typename OP, typename... Args>
__global__ void gpu_kernel(int N, Args... args) {
  CUDA_KERNEL_LOOP(idx, N) {
    OP::Map(idx, args...);
  }  
}

template<typename OP, class Context>
class GpuKernelLauncher {
 public:
  template<typename... Args>
  inline static void Launch(int N, const Context& context, Args... args) {
    int nblock = GetThreadsNum(N);
    int ngrid = GetBlockNum(CUDA_GET_BLOCKS(N, nblock));
    cudaStream_t stream = context.cuda_stream(); 
    gpu_kernel<OP, Args...>
        <<<ngrid, nblock, 0, stream>>>(N, args...);
  }
};

} // namespace blaze
