/*
 * \file broadcast.h
 * \brief The broadcast utils 
 *
 * The broadcast reference:
 *
 * https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
 *
 */
#pragma once

#include "blaze/common/blob.h"
#include "blaze/common/context.h"
#include "blaze/common/exception.h"
#include "blaze/math/gemm.h"
#include "blaze/math/elementwise/elementwise_kernel.h"

namespace blaze {

// Multidirectional Broadcasting
struct MBroadcasting {
  // Get Multidirectional Broadcasting shape
  // For example:
  // 
  // shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==>
  // shape(result) = (2, 3, 4, 5)
  //
  // shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
  //
  // shape(A) = (4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4,
  // 5)
  //
  // shape(A) = (1, 4, 5), shape(B) = (2, 3, 1, 1), ==> shape(result) = (2, 3,
  // 4, 5)
  //
  // shape(A) = (3, 4, 5), shape(B) = (2, 1, 1, 1), ==> shape(result) = (2, 3,
  // 4, 5)
  //
  static inline bool BroadcastShape(const std::vector<TIndex>& a,
                                    const std::vector<TIndex>& b,
                                    std::vector<TIndex>& out) {
    if (a.size() == 0 || b.size() == 0) return false;
    if (a == b) {
      out.resize(a.size());
      memcpy(out.data(), a.data(), a.size() * sizeof(TIndex));
      return true;
    }
    size_t out_size = std::max(a.size(), b.size());
    out.resize(out_size);
    size_t a_dis = out_size - a.size();
    size_t b_dis = out_size - b.size();
    for (size_t i = 0; i < out_size; ++i) {
      TIndex a_dim = 1;
      TIndex b_dim = 1;
      if (i >= a_dis) a_dim = a[i - a_dis];
      if (i >= b_dis) b_dim = b[i - b_dis];
      if (a_dim != b_dim) {
        if (!(a_dim == 1 || b_dim == 1)) {
          return false;
        }
        out[i] = std::max(a_dim, b_dim);
      } else {
        out[i] = a_dim;
      }
    }
    return true;
  }
};

// Unidirectional Broadcasting
struct UBroadcasting {
  // Get Unidirectional Broadcasting shape
  // For example:
  //
  // shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==>
  // shape(result) = (2, 3, 4, 5)
  //
  // shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
  //
  // shape(A) = (2, 3, 4, 5), shape(B) = (2, 1, 1, 5), ==> shape(result) = (2,
  // 3, 4, 5)
  //
  // shape(A) = (2, 3, 4, 5), shape(B) = (1, 3, 1, 5), ==> shape(result) = (2,
  // 3, 4, 5)
  //
  static inline bool BroadcastShape(const std::vector<TIndex>& a,
                                    const std::vector<TIndex>& b,
                                    std::vector<TIndex>& out) {
    if (b.size() > a.size()) {
      return false;
    }
    return MBroadcasting::BroadcastShape(a, b, out);
  }

  // shape(A) = (2, 3, 4, 5) shape(B) = (4, 5)
  // shape(A) = (2, 3, 4, 5) shape(B) = (5)
  static inline bool DimEqual(const std::vector<TIndex>& a,
                              const std::vector<TIndex>& b) {
    if (b.size() > a.size()) {
      return DimEqual(b, a);
    }
    size_t offset = a.size() - b.size();
    bool equal = true;
    for (size_t i = 0; i < b.size(); ++i) {
      if (a[offset +i] != b[i]) {
        equal = false;
        break;
      }
    }
    return equal;
  }
}; 

// Broacast assign and fma operation.
// The x_shape is same with y_shape's subshape.
template <typename DType, class Context>
void DimEqualBroadcastAssign(DType* y,
                             const std::vector<TIndex>& y_shape,
                             const DType* x,
                             const std::vector<TIndex>& x_shape,
                             Context* ctx);
template <typename DType, class Context>
void DimEqualBroadcastFMA(DType* y,
                          const std::vector<TIndex>& y_shape,
                          const DType* x,
                          const std::vector<TIndex>& x_shape,
                          Context* ctx);

// Batched broadcast copy data, the batch count of x and y are both equals batch_count.
// Which is used in parallel GEMM bias broadcast assign etc.
template <typename DType, class Context>
void DimEqualBatchedBroadcastAssign(DType* y,
                                    int batch_count,
                                    const std::vector<TIndex>& y_shape,
                                    const DType* x,
                                    const std::vector<TIndex>& x_shape,
                                    Context* ctx);

// Broadcast gemm.
// Which is mainly used in BatchDot operator etc.
template <typename DType, class Context>
void BroadcastGemm(const CBLAS_TRANSPOSE TransA,
                   const CBLAS_TRANSPOSE TransB,
                   int M,
                   int N,
                   int K,
                   float alpha,
                   const DType* A,
                   const DType* B,
                   float beta,
                   DType* C,
                   const std::vector<TIndex>& a_shape,
                   const std::vector<TIndex>& b_shape,
                   Context* context);

template <typename DType, class Context>
inline void BroadcastGemm_NxN(const CBLAS_TRANSPOSE TransA,
                              const CBLAS_TRANSPOSE TransB,
                              int M,
                              int N,
                              int K,
                              float alpha,
                              const DType* A,
                              const DType* B,
                              float beta,
                              DType* C,
                              const std::vector<TIndex>& a_shape,
                              const std::vector<TIndex>& b_shape,
                              Context* context) {
  TIndex a_size = 1;
  for (auto dim : a_shape) a_size *= dim;
  TIndex b_size = 1;
  for (auto dim : b_shape) b_size *= dim;

  if (a_size == 1 && b_size == 1) {
    Gemm<DType, Context>(TransA,
                         TransB,
                         M,
                         N,
                         K,
                         alpha,
                         A,
                         B,
                         beta,
                         C,
                         context);
  } else if (a_size != 1 && b_size != 1) {
    BLAZE_CONDITION_THROW(a_size == b_size, "a_size=", a_size, " b_size=", b_size);
    GemmStridedBatched<DType, Context>(TransA,
                                       TransB,
                                       M,
                                       N,
                                       K,
                                       alpha,
                                       A,
                                       M * K,
                                       B,
                                       K * N,
                                       beta,
                                       C,
                                       M * N,
                                       a_size,
                                       context);
  } else {
    BLAZE_CONDITION_THROW(a_size == 1 || b_size == 1, "a_size=", a_size, " b_size=", b_size);
    GemmStridedBatched<DType, Context>(TransA,
                                       TransB,
                                       M,
                                       N,
                                       K,
                                       alpha,
                                       A,
                                       a_size == 1 ? 0 : M * K,
                                       B,
                                       b_size == 1 ? 0 : K * N,
                                       beta,
                                       C,
                                       M * N,
                                       a_size == 1 ? b_size : a_size,
                                       context);
  }
}

#ifndef INSTANTIATE_BROADCAST_GEMM
#define INSTANTIATE_BROADCAST_GEMM(T, Context)                                                       \
    template <>                                                                                      \
    void BroadcastGemm<T, Context>(const CBLAS_TRANSPOSE TransA,                                     \
                                   const CBLAS_TRANSPOSE TransB,                                     \
                                   int M,                                                            \
                                   int N,                                                            \
                                   int K,                                                            \
                                   float alpha,                                                      \
                                   const T* A,                                                       \
                                   const T* B,                                                       \
                                   float beta,                                                       \
                                   T* C,                                                             \
                                   const std::vector<TIndex>& a_shape,                               \
                                   const std::vector<TIndex>& b_shape,                               \
                                   Context* context) {                                               \
      if (a_shape.size() <= 1 && b_shape.size() <= 1) {                                              \
        BroadcastGemm_NxN(TransA, TransB, M, N, K, alpha, A, B, beta, C, a_shape, b_shape, context); \
      } else {                                                                                       \
        BLAZE_THROW("Not Implemented BroadcastGemm!");                                               \
      }                                                                                              \
    }
#endif

// Batched broadcast GEMM
// Which is used for Fused Parallel BatchDot operator etc.
template <typename DType, class Context>
void BatchedBroadcastGemm(const CBLAS_TRANSPOSE TransA,
                          const CBLAS_TRANSPOSE TransB,
                          int M,
                          int N,
                          int K,
                          float alpha,
                          const DType* A,
                          const DType* B,
                          float beta,
                          DType* C,
                          const std::vector<TIndex>& a_shape,
                          const std::vector<TIndex>& b_shape,
                          int batch_count,
                          Context* context);

template <typename DType, class Context>
inline void BatchedBroadcastGemm_NxN(const CBLAS_TRANSPOSE TransA,
                                     const CBLAS_TRANSPOSE TransB,
                                     int M,
                                     int N,
                                     int K,
                                     float alpha,
                                     const DType* A,
                                     const DType* B,
                                     float beta,
                                     DType* C,
                                     const std::vector<TIndex>& a_shape,
                                     const std::vector<TIndex>& b_shape,
                                     int batch_count,
                                     Context* context) {
  TIndex a_size = 1;
  for (auto dim : a_shape) a_size *= dim;
  TIndex b_size = 1;
  for (auto dim : b_shape) b_size *= dim;

  if (a_size == 1 && b_size == 1) {
    GemmStridedBatched<DType, Context>(TransA,
                                       TransB,
                                       M,
                                       N,
                                       K,
                                       alpha,
                                       A,
                                       M * K,
                                       B,
                                       K * N,
                                       beta,
                                       C,
                                       M * N,
                                       batch_count,
                                       context);
  } else if (a_size != 1 && b_size != 1) {
    BLAZE_CONDITION_THROW(a_size == b_size, "a_size=", a_size, " b_size=", b_size);
    GemmStridedBatched<DType, Context>(TransA,
                                       TransB,
                                       M,
                                       N,
                                       K,
                                       alpha,
                                       A,
                                       M * K,
                                       B,
                                       K * N,
                                       beta,
                                       C,
                                       M * N,
                                       a_size * batch_count,
                                       context);
  } else if (b_size > a_size && N == 1) {
    BLAZE_CONDITION_THROW(a_size == 1, "a_size=", a_size, " is not one");
    //  A: M * K
    //  B: b_size * K * N(=1)
    //  op(A) * op(B) euqals: B * Transpose(op(A)).
    GemmStridedBatched<DType, Context>(CblasNoTrans,
                                       TransA == CblasTrans ? CblasNoTrans : CblasTrans,
                                       b_size,
                                       M,
                                       K,
                                       alpha,
                                       B,
                                       b_size * K,  // new lda
                                       A,
                                       M * K,       // new ldb
                                       beta,
                                       C,
                                       b_size * M,
                                       batch_count,
                                       context);
  } else {
    if (b_size > a_size) {
      BLAZE_CONDITION_THROW(a_size == 1, "a_size=", a_size, " is not one");
      for (int i = 0; i < batch_count; ++i) {
        GemmStridedBatched<DType, Context>(TransA,
                                           TransB,
                                           M,
                                           N,
                                           K,
                                           alpha,
                                           A + M * K * i,
                                           0,
                                           B + i * (b_size * K * N),
                                           K * N,
                                           beta,
                                           C + i * (b_size * M * N),
                                           M * N,
                                           b_size,
                                           context);
      }
    } else {
      BLAZE_CONDITION_THROW(b_size == 1, "b_size=", b_size, " is not one");
      for (int i = 0; i < batch_count; ++i) {
        GemmStridedBatched<DType, Context>(TransA,
                                           TransB,
                                           M,
                                           N,
                                           K,
                                           alpha,
                                           A + i * (a_size * M * K),
                                           M * K,
                                           B + K * N * i,
                                           K * N,
                                           beta,
                                           C + i * (a_size * M * N),
                                           M * N,
                                           a_size,
                                           context
                                           );
      }
    }
  }
}

#ifndef INSTANTIATE_BATCHED_BROADCAST_GEMM
#define INSTANTIATE_BATCHED_BROADCAST_GEMM(T, Context)                                                      \
    template <>                                                                                             \
    void BatchedBroadcastGemm<T, Context>(const CBLAS_TRANSPOSE TransA,                                     \
                                          const CBLAS_TRANSPOSE TransB,                                     \
                                          int M,                                                            \
                                          int N,                                                            \
                                          int K,                                                            \
                                          float alpha,                                                      \
                                          const T* A,                                                       \
                                          const T* B,                                                       \
                                          float beta,                                                       \
                                          T* C,                                                             \
                                          const std::vector<TIndex>& a_shape,                               \
                                          const std::vector<TIndex>& b_shape,                               \
                                          int batch_count,                                                  \
                                          Context* context) {                                               \
      if (a_shape.size() <= 1 && b_shape.size() <= 1) {                                                     \
        BatchedBroadcastGemm_NxN(TransA, TransB, M, N, K, alpha, A, B, beta, C, a_shape, b_shape,           \
                                 batch_count, context);                                                     \
      } else {                                                                                              \
        BLAZE_THROW("Not Implemented BatchedBroadcastGemm!");                                               \
      }                                                                                                     \
    }
#endif

// Batched broadcast mul
template <typename DType, class Context>
void BatchedBroadcastMul(const DType* a,
                         TIndex lda,
                         const std::vector<TIndex>& a_shape,
                         const DType* b,
                         TIndex ldb,
                         const std::vector<TIndex>& b_shape,
                         DType* c,
                         TIndex ldc,
                         const std::vector<TIndex>& c_shape,
                         int batch_count,
                         Context* context);

}  // namespace blaze
