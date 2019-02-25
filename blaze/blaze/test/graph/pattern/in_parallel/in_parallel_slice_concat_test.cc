/*
 * \file in_parallel_slice_concat_test.cc
 * \brief The parallel slice concat test unit
 */
#include "gtest/gtest.h"

#include "blaze/test/graph/pattern/pattern_test_common.h"

namespace blaze {

TEST(TestInOrder, All) {
  CheckPatternOutput(
      "./utest_data/graph/pattern/in_parallel/parallel_slice_concat_fusion.blaze",
      "./utest_data/graph/pattern/in_parallel/parallel_slice_concat_fusion_expected.blaze");
}

}  // namespace blaze 

