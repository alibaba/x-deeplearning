/*
 * \file softmax_op_test.cc
 * \brief The softmax op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestSigmoid, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/sigmoid/sigmoid_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestSigmoid, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/sigmoid/sigmoid_net_v1_gpu.conf", output_str);
  TestOperatorOutput<float16>("./utest_data/operator/op/sigmoid/sigmoid_net_v2_gpu.conf", output_str);
}
#endif

}  // namespace blaze
