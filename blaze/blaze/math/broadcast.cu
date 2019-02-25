/*
 * \file broadcast.cu
 * \brief The broadcast utils 
 */
#include "blaze/math/broadcast.h"

#include "blaze/math/elementwise/broadcast_elementwise.h"
#include "blaze/math/elementwise/gpu_kernel_launcher.h"
#include "blaze/math/elementwise/elementwise_kernel.h"

namespace blaze {

template <typename DType, typename OP>
__global__ void UBroadcastUnaryDimEqualKernel(DType* y, TIndex y_size,
                                              const DType* x, TIndex x_size) {
  CUDA_KERNEL_LOOP(index, y_size) {
    y[index] = OP::Map(x[index % x_size], y[index]);
  }
}

template <typename DType, typename OP>
static void UBroadcastUnaryDimEqual(DType* y,
                                     const std::vector<TIndex>& y_shape,
                                     const DType* x,
                                     const std::vector<TIndex>& x_shape,
                                     CUDAContext* ctx) {
  TIndex y_size = 1;
  for (auto dim : y_shape) y_size *= dim;
  TIndex x_size = 1;
  for (auto dim : x_shape) x_size *= dim;

  int thread_num = GetThreadsNum(y_size);
  int block_num = GetBlockNum(CUDA_GET_BLOCKS(y_size, thread_num));
  cudaStream_t stream = ctx->cuda_stream();

  UBroadcastUnaryDimEqualKernel<DType, OP><<<block_num, thread_num, 0, stream>>>
      (y, y_size, x, x_size);
}

#ifndef INSTANTIATE_BROADCAST_UNARY
#define INSTANTIATE_BROADCAST_UNARY(FuncName, OP, T)                                       \
    template <>                                                                            \
    void FuncName<T, CUDAContext>(T* y,                                                    \
                                  const std::vector<TIndex>& y_shape,                      \
                                  const T* x,                                              \
                                  const std::vector<TIndex>& x_shape,                      \
                                  CUDAContext* ctx) {                                      \
      UBroadcastUnaryDimEqual<T, OP>(y, y_shape, x, x_shape, ctx);                         \
    }
#endif

// INSTANTIATE Broadcast ASSIGN
INSTANTIATE_BROADCAST_UNARY(DimEqualBroadcastAssign, broadcast::Assign, float16)
INSTANTIATE_BROADCAST_UNARY(DimEqualBroadcastAssign, broadcast::Assign, float)
INSTANTIATE_BROADCAST_UNARY(DimEqualBroadcastAssign, broadcast::Assign, double)

INSTANTIATE_BROADCAST_UNARY(DimEqualBroadcastFMA, broadcast::Sum, float16)
INSTANTIATE_BROADCAST_UNARY(DimEqualBroadcastFMA, broadcast::Sum, float)
INSTANTIATE_BROADCAST_UNARY(DimEqualBroadcastFMA, broadcast::Sum, double)

#undef INSTANTIATE_BROADCAST_UNARY

template <typename DType, typename OP>
__global__ void BatchedUBroadcastUnaryDimEqualKernel(DType* y, int batch_count, TIndex y_size,
                                                     const DType* x, TIndex x_size) {
  TIndex total_y_size = batch_count * y_size;
  CUDA_KERNEL_LOOP(index, total_y_size) {
    int batch_index = index / y_size;
    int batch_offset = index % y_size;
    y[index] = OP::Map(x[batch_index * x_size + batch_offset % x_size], y[index]);
  }
}

template <typename DType, typename OP>
static void BacthedUBroadcastUnaryDimEqual(DType* y,
                                           int batch_count,
                                           const std::vector<TIndex>& y_shape,
                                           const DType* x,
                                           const std::vector<TIndex>& x_shape,
                                           CUDAContext* ctx) {
  TIndex y_size = 1;
  for (auto dim : y_shape) y_size *= dim;
  TIndex x_size = 1;
  for (auto dim : x_shape) x_size *= dim;

  int thread_num = GetThreadsNum(y_size * batch_count);
  int block_num = GetBlockNum(CUDA_GET_BLOCKS(y_size * batch_count, thread_num));
  cudaStream_t stream = ctx->cuda_stream();

  BatchedUBroadcastUnaryDimEqualKernel<DType, OP><<<block_num, thread_num, 0, stream>>>
      (y, batch_count, y_size, x, x_size);
}

#ifndef INSTANTIATE_BATCHED_BROADCAST_UNARY
#define INSTANTIATE_BATCHED_BROADCAST_UNARY(FuncName, OP, T)                             \
    template <>                                                                          \
    void FuncName<T, CUDAContext>(T* y,                                                  \
                                  int batch_count,                                       \
                                  const std::vector<TIndex>& y_shape,                    \
                                  const T* x,                                            \
                                  const std::vector<TIndex>& x_shape,                    \
                                  CUDAContext* ctx) {                                    \
      BacthedUBroadcastUnaryDimEqual<T, OP>(y, batch_count, y_shape, x, x_shape, ctx);   \
    }
#endif

// INSTANTIATE Batched Broadcast UNARY
INSTANTIATE_BATCHED_BROADCAST_UNARY(DimEqualBatchedBroadcastAssign, broadcast::Assign, float16)
INSTANTIATE_BATCHED_BROADCAST_UNARY(DimEqualBatchedBroadcastAssign, broadcast::Assign, float)
INSTANTIATE_BATCHED_BROADCAST_UNARY(DimEqualBatchedBroadcastAssign, broadcast::Assign, double)

#undef INSTANTIATE_BATCHED_BROADCAST_UNARY

// INSTANTIATE Broadcast GEMM
INSTANTIATE_BROADCAST_GEMM(float16, CUDAContext)
INSTANTIATE_BROADCAST_GEMM(float, CUDAContext)
INSTANTIATE_BROADCAST_GEMM(double, CUDAContext)

// INSTANTIATE Batched Broadcast GEMM
INSTANTIATE_BATCHED_BROADCAST_GEMM(float16, CUDAContext)
INSTANTIATE_BATCHED_BROADCAST_GEMM(float, CUDAContext)
INSTANTIATE_BATCHED_BROADCAST_GEMM(double, CUDAContext)

template <typename DType, typename OP>
static void BatchedBroadcastElementwise(const DType* a,
                                        const std::vector<TIndex>& a_shape,
                                        const DType* b,
                                        const std::vector<TIndex>& b_shape,
                                        DType* c,
                                        const std::vector<TIndex>& c_shape,
                                        int batch_count,
                                        CUDAContext* ctx) {
  std::vector<TIndex> ba_shape;
  std::vector<TIndex> bb_shape;
  std::vector<TIndex> bc_shape;

  ba_shape.push_back(batch_count);
  for (size_t k = 0; k < std::max(a_shape.size(), b_shape.size()) - a_shape.size(); ++k) {
    ba_shape.push_back(1UL);
  }
  for (auto dim : a_shape) ba_shape.push_back(dim);
    
  bb_shape.push_back(batch_count);
  for (size_t k = 0; k < std::max(a_shape.size(), b_shape.size()) - b_shape.size(); ++k) {
    bb_shape.push_back(1UL);
  }
  for (auto dim : b_shape) bb_shape.push_back(dim);

  bc_shape.push_back(batch_count);
  for (auto dim : c_shape) bc_shape.push_back(dim);

  bool res = broadcast::BroadcastCompute<DType, OP, GpuKernelLauncher, CUDAContext>(a, ba_shape, b, bb_shape, c, bc_shape, *ctx);
  BLAZE_CONDITION_THROW(res, "broadcast compute failed");
}

#ifndef INSTANTIATE_BATCHED_BROADCAST_ELEMENTWISE
#define INSTANTIATE_BATCHED_BROADCAST_ELEMENTWISE(FuncName, OP, T)                                            \
    template <>                                                                                               \
    void FuncName<T, CUDAContext>(const T* a,                                                                 \
                                 TIndex lda,                                                                  \
                                 const std::vector<TIndex>& a_shape,                                          \
                                 const T* b,                                                                  \
                                 TIndex ldb,                                                                  \
                                 const std::vector<TIndex>& b_shape,                                          \
                                 T* c,                                                                        \
                                 TIndex ldc,                                                                  \
                                 const std::vector<TIndex>& c_shape,                                          \
                                 int batch_count,                                                             \
                                 CUDAContext* context) {                                                      \
      BatchedBroadcastElementwise<T, OP>(a, a_shape, b, b_shape, c, c_shape, batch_count, context);           \
    }
#endif

// INSTANTIATE Batched Broadcast Mul
INSTANTIATE_BATCHED_BROADCAST_ELEMENTWISE(BatchedBroadcastMul, broadcast::Mul, float16)
INSTANTIATE_BATCHED_BROADCAST_ELEMENTWISE(BatchedBroadcastMul, broadcast::Mul, float)
INSTANTIATE_BATCHED_BROADCAST_ELEMENTWISE(BatchedBroadcastMul, broadcast::Mul, double)

#undef INSTANTIATE_BATCHED_BROADCAST_ELEMENTWISE

}  // namespace blaze
