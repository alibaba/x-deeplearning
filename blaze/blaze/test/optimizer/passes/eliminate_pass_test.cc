/*
 * \file eliminate_pass_test.cc
 * \brief The eliminate pass test 
 */
#include "gtest/gtest.h"

#include "blaze/test/optimizer/pass_test_common.h"

namespace blaze {

TEST(TestInOrder, All) {
  CheckPassOutput(
      "./utest_data/optimizer/pass/eliminate_pass.blaze",
      "./utest_data/optimizer/pass/eliminate_pass_expected.blaze");
}

}  // namespace blaze 

