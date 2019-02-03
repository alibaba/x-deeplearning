/*
 * \file xdl_sparse_fusion_pass_test.cc
 * \brief The xdl sparse fusion pass test
 */
#include "gtest/gtest.h"

#include "blaze/test/optimizer/pass_test_common.h"

namespace blaze {

TEST(TestXdlSparseFusionPass, UniqueNodeEliminatePass) {
  CheckPassOutputWithoutLoad(
      "./utest_data/optimizer/pass/xdl_sparse_fusion_pass_1.blaze",
      "./utest_data/optimizer/pass/xdl_sparse_fusion_pass_1_expected.blaze");
}

TEST(TestXdlSparseFusionPass, PsPullNodeEliminatePass) {
  CheckPassOutputWithoutLoad(
      "./utest_data/optimizer/pass/xdl_sparse_fusion_pass_2.blaze",
      "./utest_data/optimizer/pass/xdl_sparse_fusion_pass_2_expected.blaze");
}

TEST(TestXdlSparseFusionPass, PsSparsePullNodeReplacePass) {
  CheckPassOutputWithoutLoad(
      "./utest_data/optimizer/pass/xdl_sparse_fusion_pass_3.blaze",
      "./utest_data/optimizer/pass/xdl_sparse_fusion_pass_3_expected.blaze");
}

}  // namespace blaze
