/*
 * \file softmax_op_test.cc
 * \brief The softmax op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestSoftmax, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/softmax/softmax_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestSoftmax, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/softmax/softmax_net_v1_gpu.conf", output_str);
}
#endif

}  // namespace blaze
