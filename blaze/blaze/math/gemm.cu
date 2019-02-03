/*
 * \file gemm.cu
 * \brief The gemm device kernel
 */
#include "blaze/math/gemm.h"

#include "blaze/common/common_defines.h"
#include "blaze/common/exception.h"
#include "blaze/math/float16.h"

namespace blaze {

template <>
void Gemm<float16, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB,
                                const int M,
                                const int N,
                                const int K,
                                const float alpha,
                                const float16* A,
                                const float16* B,
                                const float beta,
                                float16* C,
                                CUDAContext* ctx) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  float16 alpha_h, beta_h;
  float2half(&alpha, 1, &alpha_h);
  float2half(&beta, 1, &beta_h);

  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  // We now use cublas method for smaller matrix. NOTE: Optimization on small matrix.
  CUBLAS_CHECK(cublasHgemm(ctx->cublas_handle(),
                           cuTransB,
                           cuTransA,
                           N,
                           M,
                           K,
                           (const __half*)&alpha_h,
                           (const __half*)B,
                           ldb,
                           (const __half*)A,
                           lda,
                           (const __half*)&beta_h,
                           (__half*)C,
                           ldc)); 
}

template <>
void Gemm<float, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                              const CBLAS_TRANSPOSE TransB,
                              const int M,
                              const int N,
                              const int K,
                              const float alpha,
                              const float* A,
                              const float* B,
                              const float beta,
                              float* C,
                              CUDAContext* ctx) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  // We now use cublas method for smaller matrix. NOTE: Optimization on small matrix.
  CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle(),
                           cuTransB,
                           cuTransA,
                           N,
                           M,
                           K,
                           &alpha,
                           B,
                           ldb,
                           A,
                           lda,
                           &beta,
                           C,
                           ldc));
}

template <>
void Gemm<double, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB,
                               const int M,
                               const int N,
                               const int K,
                               const float alpha,
                               const double* A,
                               const double* B,
                               const float beta,
                               double* C,
                               CUDAContext* ctx) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  double alpha_d = alpha, beta_d = beta;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  // We now use cublas method for smaller matrix. NOTE: Optimization on small matrix.
  CUBLAS_CHECK(cublasDgemm(ctx->cublas_handle(),
                           cuTransB,
                           cuTransA,
                           N,
                           M,
                           K,
                           &alpha_d,
                           B,
                           ldb,
                           A,
                           lda,
                           &beta_d,
                           C,
                           ldc));
}

template <>
void GemmEx<float16, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                                  const CBLAS_TRANSPOSE TransB,
                                  const int M,
                                  const int N,
                                  const int K,
                                  const float alpha,
                                  const float16* A,
                                  const int lda,
                                  const float16* B,
                                  const int ldb,
                                  const float beta,
                                  float16* C,
                                  const int ldc,
                                  CUDAContext* ctx) {
  float16 alpha_h, beta_h;
  float2half(&alpha, 1, &alpha_h);
  float2half(&beta, 1, &beta_h);
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasHgemm(ctx->cublas_handle(),
                           cuTransB,
                           cuTransA,
                           N,
                           M,
                           K,
                           (const __half*)&alpha_h,
                           (const __half*)B,
                           ldb,
                           (const __half*)A,
                           lda,
                           (const __half*)&beta_h,
                           (__half*)C,
                           ldc));
}

template <>
void GemmEx<float, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB,
                                const int M,
                                const int N,
                                const int K,
                                const float alpha,
                                const float* A,
                                const int lda,
                                const float* B,
                                const int ldb,
                                const float beta,
                                float* C,
                                const int ldc,
                                CUDAContext* ctx) {
  float alpha_f = alpha, beta_f = beta;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle(),
                           cuTransB,
                           cuTransA,
                           N,
                           M,
                           K,
                           &alpha_f,
                           B,
                           ldb,
                           A,
                           lda,
                           &beta_f,
                           C,
                           ldc));
}

template <>
void GemmEx<double, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                                 const CBLAS_TRANSPOSE TransB,
                                 const int M,
                                 const int N,
                                 const int K,
                                 const float alpha,
                                 const double* A,
                                 const int lda,
                                 const double* B,
                                 const int ldb,
                                 const float beta,
                                 double* C,
                                 const int ldc,
                                 CUDAContext* ctx) {
  double alpha_d = alpha, beta_d = beta;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(ctx->cublas_handle(),
                           cuTransB,
                           cuTransA,
                           N,
                           M,
                           K,
                           &alpha_d,
                           B,
                           ldb,
                           A,
                           lda,
                           &beta_d,
                           C,
                           ldc));
}

template <>
void GemmStridedBatched<float16, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                                              const CBLAS_TRANSPOSE TransB,
                                              const int M,
                                              const int N,
                                              const int K,
                                              const float alpha,
                                              const float16* A,
                                              const long long int stride_a,
                                              const float16* B,
                                              const long long int stride_b,
                                              const float beta,
                                              float16* C,
                                              const long long int stride_c,
                                              int batch_count,
                                              CUDAContext* ctx) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  
  float16 alpha_h, beta_h;
  float2half(&alpha, 1, &alpha_h);
  float2half(&beta, 1, &beta_h);

  // We now use cublas method for smaller matrix. NOTE: Optimization on small matrix.
  CUBLAS_CHECK(cublasHgemmStridedBatched(ctx->cublas_handle(),
                                         cuTransB,
                                         cuTransA,
                                         N,
                                         M,
                                         K,
                                         (const __half*)&alpha_h,
                                         (const __half*)B,
                                         ldb,
                                         stride_b,
                                         (const __half*)A,
                                         lda,
                                         stride_a,
                                         (const __half*)&beta_h,
                                         (__half*)C,
                                         ldc,
                                         stride_c,
                                         batch_count));
}

template <>
void GemmStridedBatched<float, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                                            const CBLAS_TRANSPOSE TransB,
                                            const int M,
                                            const int N,
                                            const int K,
                                            const float alpha,
                                            const float* A,
                                            const long long int stride_a,
                                            const float* B,
                                            const long long int stride_b,
                                            const float beta,
                                            float* C,
                                            const long long int stride_c,
                                            int batch_count,
                                            CUDAContext* ctx) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  // We now use cublas method for smaller matrix. NOTE: Optimization on small matrix.
  CUBLAS_CHECK(cublasSgemmStridedBatched(ctx->cublas_handle(),
                                         cuTransB,
                                         cuTransA,
                                         N,
                                         M,
                                         K,
                                         &alpha,
                                         B,
                                         ldb,
                                         stride_b,
                                         A,
                                         lda,
                                         stride_a,
                                         &beta,
                                         C,
                                         ldc,
                                         stride_c,
                                         batch_count));
}

template <>
void GemmStridedBatched<double, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                                             const CBLAS_TRANSPOSE TransB,
                                             const int M,
                                             const int N,
                                             const int K,
                                             const float alpha,
                                             const double* A,
                                             const long long int stride_a,
                                             const double* B,
                                             const long long int stride_b,
                                             const float beta,
                                             double* C,
                                             const long long int stride_c,
                                             int batch_count,
                                             CUDAContext* ctx) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  double alpha_d = alpha, beta_d = beta;
  // We now use cublas method for smaller matrix. NOTE: Optimization on small matrix.
  CUBLAS_CHECK(cublasDgemmStridedBatched(ctx->cublas_handle(),
                                         cuTransB,
                                         cuTransA,
                                         N,
                                         M,
                                         K,
                                         &alpha_d,
                                         B,
                                         ldb,
                                         stride_b,
                                         A,
                                         lda,
                                         stride_a,
                                         &beta_d,
                                         C,
                                         ldc,
                                         stride_c,
                                         batch_count));
}

template <>
void GemmBatched<float16, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                                       const CBLAS_TRANSPOSE TransB,
                                       const int M,
                                       const int N,
                                       const int K,
                                       const float alpha,
                                       const float16* A_array[],
                                       const float16* B_array[],
                                       const float beta,
                                       float16* C_array[],
                                       int batch_count,
                                       CUDAContext* ctx) {
#if CUDA_VERSION >= 9000  // Only CUDA9.0 Can support
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  
  float16 alpha_h, beta_h;
  float2half(&alpha, 1, &alpha_h);
  float2half(&beta, 1, &beta_h);

  // We now use cublas method for smaller matrix. NOTE: Optimization on small matrix.
  CUBLAS_CHECK(cublasHgemmBatched(ctx->cublas_handle(),
                                  cuTransB,
                                  cuTransA,
                                  N,
                                  M,
                                  K,
                                  (const __half*)&alpha_h,
                                  (const __half**)B_array,
                                  ldb,
                                  (const __half**)A_array,
                                  lda,
                                  (const __half*)&beta_h,
                                  (__half**)C_array,
                                  ldc,
                                  batch_count));
#else
  BLAZE_THROW("Not supported, CUDA_VERSION < 9000");
#endif
}

template <>
void GemmBatched<float, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                                     const CBLAS_TRANSPOSE TransB,
                                     const int M,
                                     const int N,
                                     const int K,
                                     const float alpha,
                                     const float* A_array[],
                                     const float* B_array[],
                                     const float beta,
                                     float* C_array[],
                                     int batch_count,
                                     CUDAContext* ctx) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  // We now use cublas method for smaller matrix. NOTE: Optimization on small matrix.
  CUBLAS_CHECK(cublasSgemmBatched(ctx->cublas_handle(),
                                  cuTransB,
                                  cuTransA,
                                  N,
                                  M,
                                  K,
                                  &alpha,
                                  B_array,
                                  ldb,
                                  A_array,
                                  lda,
                                  &beta,
                                  C_array,
                                  ldc,
                                  batch_count));
}

template <>
void GemmBatched<double, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                                      const CBLAS_TRANSPOSE TransB,
                                      const int M,
                                      const int N,
                                      const int K,
                                      const float alpha,
                                      const double* A_array[],
                                      const double* B_array[],
                                      const float beta,
                                      double* C_array[],
                                      int batch_count,
                                      CUDAContext* ctx) {
  double alpha_d = alpha, beta_d = beta;
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  // We now use cublas method for smaller matrix. NOTE: Optimization on small matrix.
  CUBLAS_CHECK(cublasDgemmBatched(ctx->cublas_handle(),
                                  cuTransB,
                                  cuTransA,
                                  N,
                                  M,
                                  K,
                                  &alpha_d,
                                  B_array,
                                  ldb,
                                  A_array,
                                  lda,
                                  &beta_d,
                                  C_array,
                                  ldc,
                                  batch_count));
}


template <>
void Gemv<float16, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                                const int M,
                                const int N,
                                const float alpha,
                                const float16* A,
                                const float16* x,
                                const float beta,
                                float16* y,
                                CUDAContext* ctx) {
  BLAZE_THROW("Not implemented!");
}

template <>
void Gemv<float, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                              const int M,
                              const int N,
                              const float alpha,
                              const float* A,
                              const float* x,
                              const float beta,
                              float* y,
                              CUDAContext* ctx) {
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(ctx->cublas_handle(),
                           cuTransA,
                           N,
                           M,
                           &alpha,
                           A,
                           N,
                           x,
                           1,
                           &beta,
                           y,
                           1));
}

template <>
void Gemv<double, CUDAContext>(const CBLAS_TRANSPOSE TransA,
                               const int M,
                               const int N,
                               const float alpha,
                               const double* A,
                               const double* x,
                               const float beta,
                               double* y,
                               CUDAContext* ctx) {
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  double alpha_d = alpha, beta_d = beta;
  CUBLAS_CHECK(cublasDgemv(ctx->cublas_handle(),
                           cuTransA,
                           N,
                           M,
                           &alpha_d,
                           A,
                           N,
                           x,
                           1,
                           &beta_d,
                           y,
                           1));
}

}  // namespace blaze

