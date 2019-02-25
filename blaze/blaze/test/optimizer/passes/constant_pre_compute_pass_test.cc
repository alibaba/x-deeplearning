/*
 * \file constant_pre_compute_pass_test.cc
 * \brief The constant pre compute pass test 
 */
#include "gtest/gtest.h"

#include "blaze/test/optimizer/pass_test_common.h"

namespace blaze {

TEST(TestConstantPreCompute, All) {
  CheckPassOutput(
      "./utest_data/optimizer/pass/constant_pre_compute_pass.blaze",
      "./utest_data/optimizer/pass/constant_pre_compute_pass_expected.blaze");

  CheckPassOutput(
      "./utest_data/optimizer/pass/constant_pre_compute_pass_2.blaze",
      "./utest_data/optimizer/pass/constant_pre_compute_pass_2_expected.blaze");
}

}  // namespace blaze 
