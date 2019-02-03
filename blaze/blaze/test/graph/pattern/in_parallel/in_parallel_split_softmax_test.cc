/*
 * \file in_parallel_split_softmax_test.cc
 * \brief The parallel gemm test unit, which only support Fix-sized bacthed gemm
 */
#include "gtest/gtest.h"

#include "blaze/test/graph/pattern/pattern_test_common.h"

namespace blaze {

TEST(TestInOrder, All) {
  CheckPatternOutput(
      "./utest_data/graph/pattern/in_parallel/parallel_split_softmax_fusion.blaze",
      "./utest_data/graph/pattern/in_parallel/parallel_split_softmax_fusion_expected.blaze");
}

}  // namespace blaze 

