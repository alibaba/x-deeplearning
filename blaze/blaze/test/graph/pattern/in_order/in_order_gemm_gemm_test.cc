/*
 * \file in_order_gemm_gemm_test.cc
 * \brief The in order gemm gemm test unit
 */
#include "gtest/gtest.h"

#include "blaze/graph/fusion_pattern.h"
#include "blaze/test/graph/pattern/pattern_test_common.h"

namespace blaze {

TEST(TestInOrder, All) {
  // (w0) (w1)
  CheckPatternOutput(
      "./utest_data/graph/pattern/in_order/in_order_gemm_gemm_fusion.blaze",
      "./utest_data/graph/pattern/in_order/in_order_gemm_gemm_fusion_expected.blaze");
  // (w0, bias0) (w1)
  CheckPatternOutput(
      "./utest_data/graph/pattern/in_order/in_order_gemm_gemm_fusion2.blaze",
      "./utest_data/graph/pattern/in_order/in_order_gemm_gemm_fusion2_expected.blaze");
  // (w0), (w1, bias1)
  CheckPatternOutput(
      "./utest_data/graph/pattern/in_order/in_order_gemm_gemm_fusion3.blaze",
      "./utest_data/graph/pattern/in_order/in_order_gemm_gemm_fusion3_expected.blaze");
  // (w0, bias0), (w1, bias1)
  CheckPatternOutput(
      "./utest_data/graph/pattern/in_order/in_order_gemm_gemm_fusion4.blaze",
      "./utest_data/graph/pattern/in_order/in_order_gemm_gemm_fusion4_expected.blaze");
}

}  // namespace blaze

