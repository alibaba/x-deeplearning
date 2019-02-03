/*
 * \file gemm_test.cc
 * \brief The gemm test unit
 */
#include "gtest/gtest.h"

#include "blaze/math/gemm.h"

namespace blaze {

#define M 10
#define N 6
#define K 8

TEST(TestGemm, Gemm) {
  float A[M * K];
  float B[K * N];
  float C[M * N];

  for (int i = 0; i < M * K; ++i) {
    A[i] = 1;
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = 1;
  }
  Gemm<float, CPUContext>(CblasNoTrans,
                          CblasNoTrans,
                          M,
                          N,
                          K,
                          1.0,
                          A,
                          B,
                          0,
                          C,
                          nullptr);
  for (int i = 0; i < M * N; ++i) {
    EXPECT_FLOAT_EQ(C[i], static_cast<float>(K));
  }
}

TEST(TestGemmEx, GemmEx) {
  // TODO
}

TEST(TestGemmStridedBatched, GemmStridedBatched) {
  // TODO
}

TEST(TestGemv, Gemv) {
  // TODO
}

}  // namespace blaze

