/*
 * \file in_order_slice_slice_test.cc
 * \brief The in order slice slice test unit
 */
#include "gtest/gtest.h"

#include "blaze/graph/fusion_pattern.h"
#include "blaze/test/graph/pattern/pattern_test_common.h"

namespace blaze {

TEST(TestInOrder, All) {
  CheckPatternOutput(
      "./utest_data/graph/pattern/in_order/in_order_slice_slice_fusion.blaze",
      "./utest_data/graph/pattern/in_order/in_order_slice_slice_fusion_expected.blaze");
}

}  // namespace blaze

