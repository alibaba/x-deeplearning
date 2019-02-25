/*
 * \file broadcast.cc
 * \brief The broadcast utils 
 */
#include "blaze/math/broadcast.h"

#include "blaze/common/types.h"
#include "blaze/math/elementwise/broadcast_elementwise.h"
#include "blaze/math/elementwise/cpu_kernel_launcher.h"
#include "blaze/math/elementwise/elementwise_kernel.h"

namespace blaze {

template <typename DType, typename OP>
static void UBroadcastUnaryDimEqual(DType* y,
                                    const std::vector<TIndex>& y_shape,
                                    const DType* x,
                                    const std::vector<TIndex>& x_shape) {
  TIndex y_size = 1;
  for (auto dim : y_shape) y_size *= dim;
  TIndex x_size = 1;
  for (auto dim : x_shape) x_size *= dim;
  
  for (size_t y_offset = 0; y_offset < y_size; y_offset += x_size) {
    for (size_t z = 0; z < x_size; ++z) {
      y[y_offset + z] = OP::Map(x[z], y[y_offset + z]);
    }
  }
}

#ifndef INSTANTIATE_BROADCAST_UNARY
#define INSTANTIATE_BROADCAST_UNARY(FuncName, OP, T)                                     \
    template <>                                                                          \
    void FuncName<T, CPUContext>(T* y,                                                   \
                                 const std::vector<TIndex>& y_shape,                     \
                                 const T* x,                                             \
                                 const std::vector<TIndex>& x_shape,                     \
                                 CPUContext* ctx) {                                      \
      UBroadcastUnaryDimEqual<T, OP>(y, y_shape, x, x_shape);                            \
    }
#endif

// INSTANTIATE Broadcast ASSIGN
INSTANTIATE_BROADCAST_UNARY(DimEqualBroadcastAssign, broadcast::Assign, float)
INSTANTIATE_BROADCAST_UNARY(DimEqualBroadcastAssign, broadcast::Assign, double)

INSTANTIATE_BROADCAST_UNARY(DimEqualBroadcastFMA, broadcast::Sum, float)
INSTANTIATE_BROADCAST_UNARY(DimEqualBroadcastFMA, broadcast::Sum, double)

#undef INSTANTIATE_BROADCAST_UNARY

template <typename DType, typename OP>
static void BacthedUBroadcastUnaryDimEqual(DType* y,
                                           int batch_count,
                                           const std::vector<TIndex>& y_shape,
                                           const DType* x,
                                           const std::vector<TIndex>& x_shape) {
  TIndex y_size = 1;
  for (auto dim : y_shape) y_size *= dim;
  TIndex x_size = 1;
  for (auto dim : x_shape) x_size *= dim;

  TIndex y_offset = 0, x_offset = 0;
  for (int i = 0; i < batch_count; ++i) {
    UBroadcastUnaryDimEqual<DType, OP>(y + y_offset, y_shape, x + x_offset, x_shape);
    y_offset += y_size;
    x_offset += x_size;
  }
}

#ifndef INSTANTIATE_BATCHED_BROADCAST_UNARY
#define INSTANTIATE_BATCHED_BROADCAST_UNARY(FuncName, OP, T)                            \
    template <>                                                                         \
    void FuncName<T, CPUContext>(T* y,                                                  \
                                 int batch_count,                                       \
                                 const std::vector<TIndex>& y_shape,                    \
                                 const T* x,                                            \
                                 const std::vector<TIndex>& x_shape,                    \
                                 CPUContext* ctx) {                                     \
      BacthedUBroadcastUnaryDimEqual<T, OP>(y, batch_count, y_shape, x, x_shape);       \
    }
#endif

// INSTANTIATE Batched Broadcast ASSIGN
INSTANTIATE_BATCHED_BROADCAST_UNARY(DimEqualBatchedBroadcastAssign, broadcast::Assign, float)
INSTANTIATE_BATCHED_BROADCAST_UNARY(DimEqualBatchedBroadcastAssign, broadcast::Assign, double)

// INSTANTIATE Broadcast GEMM
INSTANTIATE_BROADCAST_GEMM(float, CPUContext)
INSTANTIATE_BROADCAST_GEMM(double, CPUContext)

// INSTANTIATE Batched Broadcast GEMM
INSTANTIATE_BATCHED_BROADCAST_GEMM(float, CPUContext)
INSTANTIATE_BATCHED_BROADCAST_GEMM(double, CPUContext)

// a_shape.size() <= 3 && b_shape.size() <= 3
template <typename DType, typename OP>
static void BatchedBroadcastElementwise_3x3_Attention(const DType* a,
                                                      TIndex lda,
                                                      const std::vector<TIndex>& a_shape,
                                                      const DType* b,
                                                      TIndex ldb,
                                                      const std::vector<TIndex>& b_shape,
                                                      DType* c,
                                                      TIndex ldc,
                                                      int batch_count) {
  // User X Ad Attention.
  TIndex lhs_len = a_shape[1] * a_shape[2];
  TIndex rhs_len = b_shape[1] * b_shape[2];
  while (batch_count-- > 0) {
    for (TIndex i = 0; i < b_shape[0]; ++i) {
      const DType* l = a;
      const DType* r = b + i * rhs_len;
      DType* cur_c = c + i * lhs_len;
      for (TIndex m = 0; m < b_shape[1]; ++m) {
        DType alpha = r[m];
        // auto-vectorized.
        for (TIndex n = 0; n < a_shape[2]; ++n) {
          *cur_c++ = OP::Map(*l++, alpha);
        }
      }
    }
    a += lda; b += ldb; c += ldc;
  }
}

template <typename DType, typename OP>
static void BatchedBroadcastElementwise(const DType* a,
                                        const std::vector<TIndex>& a_shape,
                                        const DType* b,
                                        const std::vector<TIndex>& b_shape,
                                        DType* c,
                                        const std::vector<TIndex>& c_shape,
                                        int batch_count,
                                        CPUContext* ctx) {
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

  bool res = broadcast::BroadcastCompute<DType, OP, CpuKernelLauncher, CPUContext>(a, ba_shape, b, bb_shape, c, bc_shape, *ctx);
  BLAZE_CONDITION_THROW(res, "broadcast compute failed");
}

template <typename DType, typename OP>
static void BatchedBroadcastElementwise_3x3(const DType* a,
                                            TIndex lda,
                                            const std::vector<TIndex>& a_shape,
                                            const DType* b,
                                            TIndex ldb,
                                            const std::vector<TIndex>& b_shape,
                                            DType* c,
                                            const std::vector<TIndex>& c_shape,
                                            TIndex ldc,
                                            int batch_count, 
                                            CPUContext* ctx) {
  // Step1: Regen lshape and rshape
  std::vector<TIndex> l_shape = a_shape;
  std::vector<TIndex> r_shape = b_shape;
  for (size_t k = r_shape.size(); k < 3; ++k) r_shape.insert(r_shape.begin(), 1);
  for (size_t k = l_shape.size(); k < 3; ++k) l_shape.insert(l_shape.begin(), 1);
  // Step2: Special process(Used in Attention)
  if (l_shape[0] < r_shape[0] &&
      l_shape[1] == r_shape[1] &&
      l_shape[2] > r_shape[2]) {
    BatchedBroadcastElementwise_3x3_Attention<DType, OP>(a, lda, l_shape, b, ldb, r_shape, c, ldc, batch_count);
  } else if (l_shape[0] > r_shape[0] &&
             l_shape[1] == r_shape[1] &&
             l_shape[2] < r_shape[2]) {
    BatchedBroadcastElementwise_3x3_Attention<DType, OP>(b, ldb, r_shape, a, lda, l_shape, c, ldc, batch_count);
  } else {
    BatchedBroadcastElementwise<DType, OP>(a, a_shape, b, b_shape, c, c_shape, batch_count, ctx);
  }
}

#ifndef INSTANTIATE_BATCHED_BROADCAST_ELEMENTWISE
#define INSTANTIATE_BATCHED_BROADCAST_ELEMENTWISE(FuncName, OP, T)                                            \
    template <>                                                                                               \
    void FuncName<T, CPUContext>(const T* a,                                                                  \
                                 TIndex lda,                                                                  \
                                 const std::vector<TIndex>& a_shape,                                          \
                                 const T* b,                                                                  \
                                 TIndex ldb,                                                                  \
                                 const std::vector<TIndex>& b_shape,                                          \
                                 T* c,                                                                        \
                                 TIndex ldc,                                                                  \
                                 const std::vector<TIndex>& c_shape,                                          \
                                 int batch_count,                                                             \
                                 CPUContext* context) {                                                       \
      if (a_shape.size() <= 3 && b_shape.size() <= 3) {                                                       \
        BatchedBroadcastElementwise_3x3<T, OP>(a, lda, a_shape, b, ldb, b_shape, c, c_shape, ldc, batch_count, context); \
      } else {                                                                                                \
        BatchedBroadcastElementwise<T, OP>(                                                               \
                  a, a_shape, b, b_shape, c, c_shape, batch_count, context);                                  \
      }                                                                                                       \
    }
#endif

// INSTANTIATE Batched Broadcast Mul
INSTANTIATE_BATCHED_BROADCAST_ELEMENTWISE(BatchedBroadcastMul, broadcast::Mul, float)
INSTANTIATE_BATCHED_BROADCAST_ELEMENTWISE(BatchedBroadcastMul, broadcast::Mul, double)

#undef INSTANTIATE_BATCHED_BROADCAST_ELEMENTWISE

}  // namespace blaze
