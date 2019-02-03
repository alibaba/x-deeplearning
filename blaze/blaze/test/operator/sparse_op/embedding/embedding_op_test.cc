/*
 * \file embedding_op_test.cc
 * \brief The embedding op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"
#include "blaze/store/mock_sparse_puller.h"

namespace blaze {
namespace store {

SparsePuller *CreateMockSparsePuller() {
  return new MockSparsePuller();
}

REGISTER_SPARSE_PULLER_CREATION("mock_sparse_puller", CreateMockSparsePuller);

TEST(TestEmbedding, TestCPU) {
  std::vector<std::string> output_str;
  output_str.resize(2);
  output_str[0] = "block1_values";
  output_str[1] = "block2_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/embedding/embedding_v1.conf", output_str,
                            "", "mock_sparse_puller");
}

}  // namespace store
}  // namesgpace blaze
