/*
 * \file concat_reduce_swap_pass_test.cc
 * \brief The concat reduce swap pass test 
 */
#include "gtest/gtest.h"

#include "blaze/test/optimizer/pass_test_common.h"

namespace blaze {

TEST(TestConcatReduceSwapPass, All) {
  CheckPassOutput(
      "./utest_data/optimizer/pass/concat_reduce_swap_pass.blaze",
      "./utest_data/optimizer/pass/concat_reduce_swap_pass_expected.blaze");
}

}  // namespace blaze 
