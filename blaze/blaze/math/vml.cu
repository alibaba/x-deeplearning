/*
 * \file vml.cu
 * \brief The VML routine on GPU Architecture
 */
#include "blaze/math/vml.h"

#include "blaze/common/common_defines.h"
#include "blaze/math/float16.h"

namespace blaze {

template <typename T>
__global__ void VML_Set_Kernel(const int N, T* a, T v) { 
  CUDA_KERNEL_LOOP(index, N) {
    a[index] = v;
  }
}

#ifndef DECLARE_VML_SET_FUNCTION_IMPL
#define DECLARE_VML_SET_FUNCTION_IMPL(DType)                                            \
  template <>                                                                           \
  void VML_Set<DType, CUDAContext>(const int N, DType* a, DType v, CUDAContext* ctx) {  \
    int thread_num = GetThreadsNum(N);                                                  \
    int block_num = CUDA_GET_BLOCKS(N, thread_num);                                     \
    cudaStream_t stream = ctx->cuda_stream();                                           \
    VML_Set_Kernel<DType><<<block_num, thread_num, 0, stream>>>(N, a, v);               \
  }
#endif

DECLARE_VML_SET_FUNCTION_IMPL(float16)
DECLARE_VML_SET_FUNCTION_IMPL(float)
DECLARE_VML_SET_FUNCTION_IMPL(double)

#undef DECLARE_VML_SET_FUNCTION_IMPL

template <typename DstT, typename SrcT>
__global__ void VML_Set2_Kernel(const int N, DstT* a, const SrcT* b) { 
  CUDA_KERNEL_LOOP(index, N) {
    a[index] = b[index];
  }
}

#ifndef DECLARE_VML_SET2_FUNCTION_IMPL
#define DECLARE_VML_SET2_FUNCTION_IMPL(DstT, SrcT)                                                \
  template <>                                                                                     \
  void VML_Set<DstT, SrcT, CUDAContext>(const int N, DstT* a, const SrcT* b, CUDAContext* ctx) {  \
    int thread_num = GetThreadsNum(N);                                                            \
    int block_num = CUDA_GET_BLOCKS(N, thread_num);                                               \
    cudaStream_t stream = ctx->cuda_stream();                                                     \
    VML_Set2_Kernel<DstT, SrcT><<<block_num, thread_num, 0, stream>>>(N, a, b);                   \
  }
#endif

DECLARE_VML_SET2_FUNCTION_IMPL(float, float16)
DECLARE_VML_SET2_FUNCTION_IMPL(float16, float)
DECLARE_VML_SET2_FUNCTION_IMPL(float, float)
DECLARE_VML_SET2_FUNCTION_IMPL(float16, float16)

#undef DECLARE_VML_SET2_FUNCTION_IMPL

}  // namespace blaze

