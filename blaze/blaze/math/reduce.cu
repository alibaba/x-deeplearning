/*
 * \file reduce.cu
 * \brief The reduce device kernel
 */
#include "blaze/math/reduce.h"

#include "blaze/common/common_defines.h"
#include "blaze/math/float16.h"

namespace blaze {

template <typename DType, int blockDimX>
__global__ void ReduceSumKernel(const DType* x,
                                const int outer_size,
                                const int dim,
                                const int inner_size,
                                DType* y) {
  SharedMemory<DType> smem;
  DType* sdata = smem.GetPointer();

  int blockDimY = blockDim.x / blockDimX;
  int tid = threadIdx.x % blockDimX;
  int tidY = threadIdx.x / blockDimX + blockIdx.y * blockDimY;

  int bdimY = blockDimY * gridDim.y;

  // during task schedule, threadIdx.x is more preferred than threadIdx.y
  for (int lxIdx = tidY; lxIdx < inner_size; lxIdx += bdimY) {
    // process each inner_block
    int outer_id = blockIdx.x;
    for (; outer_id < outer_size; outer_id += gridDim.x) {
      DType* dst = y + outer_id * inner_size;
      const DType* src = x + outer_id * dim * inner_size + lxIdx;
      int i = tid;
      DType tmp = 0;
      // is not efficiently.
      while (i < dim) {
        tmp += src[i * inner_size];
        i += blockDimX;
      }
      sdata[threadIdx.x] = tmp;
      __syncthreads();
      if (blockDimX >= 512) {
        if (tid < 256) { sdata[threadIdx.x] += sdata[threadIdx.x + 256]; }
        __syncthreads();
      }
      if (blockDimX >= 256) {
        if (tid < 128) { sdata[threadIdx.x] += sdata[threadIdx.x + 128]; }
        __syncthreads();
      }
      if (blockDimX >= 128) {
        if (tid < 64) { sdata[threadIdx.x] += sdata[threadIdx.x + 64]; }
        __syncthreads();
      }
      if (blockDimX >= 64) {
        if (tid < 32) { sdata[threadIdx.x] += sdata[threadIdx.x + 32]; }
        __syncthreads();
      }
      // this part has to run in the same warp, which can avoid __syncthreads
      if (blockDimX >= 32) {
        if (tid < 16) sdata[threadIdx.x] += sdata[threadIdx.x + 16];
        __syncwarp();
      }
      if (blockDimX >= 16) {
        if (tid < 8) sdata[threadIdx.x] += sdata[threadIdx.x + 8];
        __syncwarp();
      }
      if (blockDimX >= 8) {
        if (tid < 4) sdata[threadIdx.x] += sdata[threadIdx.x + 4];
        __syncwarp();
      }
      if (blockDimX >= 4) {
        if (tid < 2) sdata[threadIdx.x] += sdata[threadIdx.x + 2];
        __syncwarp();
      }
      if (blockDimX >= 2) {
        if (tid < 1) sdata[threadIdx.x] += sdata[threadIdx.x + 1];
        __syncwarp();
      }
      if (tid == 0) {
        dst[lxIdx] = sdata[threadIdx.x];
        //__syncwarp();
      }
    }
  }
}

template <typename DType>
inline void ReduceSumImpl(const DType* x,
                          const int outer_size,  // 1
                          const int dim,  // 150
                          const int inner_size,  // 200
                          DType* y,
                          CUDAContext* ctx) {
  int threadsX = GetThreadsNum(dim, true);
  int threadsY = GetThreadsNum((inner_size * dim > 512 ? 512 : inner_size * dim) / threadsX, true); 
  if (threadsX * threadsY > 1024) threadsY = 1024 / threadsX;

  int blocksX = CUDA_GET_BLOCKS(outer_size, 16);
  int blocksY = CUDA_GET_BLOCKS(inner_size, threadsY);

  SWITCH_BLOCKDIM(threadsX,
                  DType,
                  ReduceSumKernel,
                  dim3(blocksX, blocksY),
                  threadsX * threadsY,
                  AlignN((sizeof(DType) * threadsX * threadsY), sizeof(DType)),
                  ctx->cuda_stream(),
                  x,
                  outer_size,
                  dim,
                  inner_size,
                  y
                  );
}

INSTANTIATE_REDUCESUM(float16, CUDAContext)
INSTANTIATE_REDUCESUM(float, CUDAContext)
INSTANTIATE_REDUCESUM(double, CUDAContext)

}  // namespace blaze
