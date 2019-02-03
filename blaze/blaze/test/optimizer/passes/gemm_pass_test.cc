/*
 * \file gemm_pass_test.cc
 * \brief The gemm pass test 
 */
#include "gtest/gtest.h"

#include "blaze/test/optimizer/pass_test_common.h"

namespace blaze {

TEST(TestInOrder, All) {
  CheckPassOutputWithoutLoad(
      "./utest_data/optimizer/pass/gemm_pass.blaze",
      "./utest_data/optimizer/pass/gemm_pass_expected.blaze");
}

}  // namespace blaze 

