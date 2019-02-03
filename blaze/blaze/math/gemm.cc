/*
 * \file gemm.cc
 * \brief The matmul device kernel.
 */
#include "blaze/math/gemm.h"

#include "blaze/common/exception.h"
#include "blaze/math/float16.h"

namespace blaze {

template <>
void Gemm<float, CPUContext>(const CBLAS_TRANSPOSE TransA,
                             const CBLAS_TRANSPOSE TransB,
                             const int M,
                             const int N,
                             const int K,
                             const float alpha,
                             const float* A,
                             const float* B,
                             const float beta,
                             float* C,
                             CPUContext* ctx) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <>
void Gemm<double, CPUContext>(const CBLAS_TRANSPOSE TransA,
                              const CBLAS_TRANSPOSE TransB,
                              const int M,
                              const int N,
                              const int K,
                              const float alpha,
                              const double* A,
                              const double* B,
                              const float beta,
                              double* C,
                              CPUContext* ctx) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <>
void GemmEx<float, CPUContext>(const CBLAS_TRANSPOSE TransA,
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
                               CPUContext* context) {
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
void GemmEx<double, CPUContext>(const CBLAS_TRANSPOSE TransA,
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
                                CPUContext* context) {
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
void GemmStridedBatched<float, CPUContext>(const CBLAS_TRANSPOSE TransA,
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
                                           CPUContext* ctx) {
#ifdef USE_MKL
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  std::vector<const float*> a_array(batch_count, nullptr);
  std::vector<const float*> b_array(batch_count, nullptr);
  std::vector<float*> c_array(batch_count, nullptr);
  for (int i = 0; i < batch_count; ++i) {
    a_array[i] = A + stride_a * i;
    b_array[i] = B + stride_b * i;
    c_array[i] = C + stride_c * i;
  }
  cblas_sgemm_batch(CblasRowMajor,
                    &TransA,
                    &TransB,
                    &M,
                    &N,
                    &K,
                    &alpha,
                    a_array.data(),
                    &lda,
                    b_array.data(),
                    &ldb,
                    &beta,
                    c_array.data(),
                    &N,
                    1,
                    &batch_count);
#else
  for (int i = 0; i < batch_count; ++i) {
    Gemm<float, CPUContext>(TransA,
                            TransB,
                            M,
                            N,
                            K,
                            alpha,
                            A + stride_a * i,
                            B + stride_b * i,
                            beta,
                            C + stride_c * i,
                            ctx);
  }
#endif
}

template <>
void GemmStridedBatched<double, CPUContext>(const CBLAS_TRANSPOSE TransA,
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
                                            CPUContext* ctx) {
#ifdef USE_MKL
  double beta_d = beta, alpha_d = alpha;
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  std::vector<const double*> a_array(batch_count, nullptr);
  std::vector<const double*> b_array(batch_count, nullptr);
  std::vector<double*> c_array(batch_count, nullptr);
  for (int i = 0; i < batch_count; ++i) {
    a_array[i] = A + stride_a * i;
    b_array[i] = B + stride_b * i;
    c_array[i] = C + stride_c * i;
  }
  cblas_dgemm_batch(CblasRowMajor,
                    &TransA,
                    &TransB,
                    &M,
                    &N,
                    &K,
                    &alpha_d,
                    a_array.data(),
                    &lda,
                    b_array.data(),
                    &ldb,
                    &beta_d,
                    c_array.data(),
                    &N,
                    1,
                    &batch_count);
#else
  for (int i = 0; i < batch_count; ++i) {
    Gemm<double, CPUContext>(TransA,
                             TransB,
                             M,
                             N,
                             K,
                             alpha,
                             A + stride_a * i,
                             B + stride_b * i,
                             beta,
                             C + stride_c * i,
                             ctx);
  }
#endif
}

template <>
void GemmBatched<float, CPUContext>(const CBLAS_TRANSPOSE TransA,
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
                                    CPUContext* ctx) {
#ifdef USE_MKL
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm_batch(CblasRowMajor,
                    &TransA,
                    &TransB,
                    &M,
                    &N,
                    &K,
                    &alpha,
                    A_array,
                    &lda,
                    B_array,
                    &ldb,
                    &beta,
                    C_array,
                    &N,
                    1,
                    &batch_count);
#else
  for (int i = 0; i < batch_count; ++i) {
    Gemm<float, CPUContext>(TransA,
                            TransB,
                            M,
                            N,
                            K,
                            alpha,
                            A_array[i],
                            B_array[i],
                            beta,
                            C_array[i],
                            ctx);
  }
#endif
}

template <>
void GemmBatched<double, CPUContext>(const CBLAS_TRANSPOSE TransA,
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
                                     CPUContext* ctx) {

#ifdef USE_MKL
  double beta_d = beta, alpha_d = alpha;
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm_batch(CblasRowMajor,
                    &TransA,
                    &TransB,
                    &M,
                    &N,
                    &K,
                    &alpha_d,
                    A_array,
                    &lda,
                    B_array,
                    &ldb,
                    &beta_d,
                    C_array,
                    &N,
                    1,
                    &batch_count);
#else
  for (int i = 0; i < batch_count; ++i) {
    Gemm<double, CPUContext>(TransA,
                             TransB,
                             M,
                             N,
                             K,
                             alpha,
                             A_array[i],
                             B_array[i],
                             beta,
                             C_array[i],
                             ctx);
  }
#endif
}

template <>
void Gemv<float, CPUContext>(const CBLAS_TRANSPOSE TransA,
                             const int M,
                             const int N,
                             const float alpha,
                             const float* A,
                             const float* x,
                             const float beta,
                             float* y,
                             CPUContext* ctx) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void Gemv<double, CPUContext>(const CBLAS_TRANSPOSE TransA,
                              const int M,
                              const int N,
                              const float alpha,
                              const double* A,
                              const double* x,
                              const float beta,
                              double* y,
                              CPUContext* ctx) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

}  // namespace blaze

