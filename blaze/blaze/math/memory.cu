/*
 * \file memory.cu
 * \brief memeory utils for cuda 
 */

#include "blaze/math/memory.h"
#include "blaze/common/common_defines.h"
#include "blaze/common/context.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename DType>
__global__ void SliceMemcpyKernel(DType* dst, size_t dpitch,
    const DType* src, size_t spitch,
    const SliceParam* slice_param,
    size_t count, size_t height) {
  DType* cur_dst;
  const DType* cur_src;
  CUDA_KERNEL_LOOP(index, height) {
    for (int i = blockIdx.y * blockDim.y + threadIdx.y;
        i < count; i += blockDim.y * gridDim.y) {
      cur_dst = dst + dpitch * index + slice_param[i].dst_idx;
      cur_src = src + spitch * index + slice_param[i].src_idx;
      int step_size = slice_param[i].step_size;
      int strip = sizeof(DType) / sizeof(DType);
      int j = 0;
      for (; j <= step_size - strip; j += strip) {
        *(DType*)(cur_dst + j) = *(DType*)(cur_src + j);
      }
      for (; j < step_size; j++) {
        cur_dst[j] = cur_src[j];
      }
    }
  }
}

template <typename DType>
void SliceMemcpyImpl(DType* dst, size_t dpitch,
    const DType* src, size_t spitch,
    const SliceParam* slice_param,
    size_t count, size_t height,
    const CUDAContext* context) {
 uint32_t threads = std::min<uint32_t>(CUDA_NUM_THREADS, height);
  cudaStream_t stream = context->cuda_stream();
  SliceMemcpyKernel <<<CUDA_GET_BLOCKS(height, threads),
                 dim3(threads, std::min<uint32_t>(count, 1024 / threads)),
                 0,
                 stream>>>(dst, dpitch, src, spitch,
                           slice_param, count, height);
}

INSTANTIATE_SLICEMEMCPY(float16, CUDAContext)
INSTANTIATE_SLICEMEMCPY(float, CUDAContext)
INSTANTIATE_SLICEMEMCPY(double, CUDAContext)

template <>
void Memcpy<CUDAContext>(void* dst, const void* src, size_t size,
    const CUDAContext* context) {
  cudaStream_t stream = context->cuda_stream();
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
}

}  // name 
