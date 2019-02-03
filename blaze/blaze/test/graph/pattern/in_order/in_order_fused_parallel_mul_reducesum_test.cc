/*
 * \file in_order_fused_parallel_mul_reducesum_test.cc
 * \brief The in order fused_parallel_mul_reducesum test unit
 */
#include "gtest/gtest.h"

#include "blaze/graph/fusion_pattern.h"
#include "blaze/test/graph/pattern/pattern_test_common.h"

namespace blaze {

TEST(TestInOrder, All) {
  CheckPatternOutput(
      "./utest_data/graph/pattern/in_order/in_order_fused_parallel_mul_reducesum_fusion.blaze",
      "./utest_data/graph/pattern/in_order/in_order_fused_parallel_mul_reducesum_fusion_expected.blaze");
}

}  // namespace blaze

