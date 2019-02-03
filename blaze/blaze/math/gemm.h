/*
 * \file gemm.h
 * \brief The blas device kernel
 */
#pragma once

#include "blaze/common/context.h"

#ifdef USE_MKL
#include "mkl.h"
#else
#include "cblas.h"
#endif

namespace blaze {

// C = alpha * op(A) * op(B) + beta * C
// A: [M, K]
// B: [K, N]
// C: [M, N]
template <typename T, class Context>
void Gemm(const CBLAS_TRANSPOSE TransA,
          const CBLAS_TRANSPOSE TransB,
          const int M,
          const int N,
          const int K,
          const float alpha,
          const T* A,
          const T* B,
          const float beta,
          T* C,
          Context* ctx);

template <typename T, class Context>
void GemmEx(const CBLAS_TRANSPOSE TransA,
            const CBLAS_TRANSPOSE TransB,
            const int M,
            const int N,
            const int K,
            const float alpha,
            const T* A,
            const int lda,
            const T* B,
            const int ldb,
            const float beta,
            T* c,
            const int ldc,
            Context* context);

template <typename T, class Context>
void GemmStridedBatched(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB,
                        const int M,
                        const int N,
                        const int K,
                        const float alpha,
                        const T* A,
                        const long long int stride_a,
                        const T* B,
                        const long long int stride_b,
                        const float beta,
                        T* C,
                        const long long int stride_c,
                        const int batch_count,
                        Context* ctx);

template <typename T, class Context>
void GemmBatched(const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB,
                 const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const T* A_array[],
                 const T* B_array[],
                 const float beta,
                 T* C_array[],
                 int batch_count,
                 Context* ctx);

// y = alpha * op(A) * x + beta * y
// A: [M, N]
template <typename T, class Context>
void Gemv(const CBLAS_TRANSPOSE TransA,
          const int M,
          const int N,
          const float alpha,
          const T* A,
          const T* x,
          const float beta,
          T* y,
          Context* contxt); 

}  // namespace blaze

