/*
 * \file prelu_op_test.cc
 * \brief The prelu op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestPRelu, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/prelu/prelu_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestPRelu, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/prelu/prelu_net_v1_gpu.conf", output_str);
}
#endif

}  // namespace blaze
