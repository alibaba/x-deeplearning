/*
 * \file not_op_test.cc
 * \brief The not op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestNot, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/not/not_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestNot, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/not/not_net_v1_gpu.conf", output_str);
}
#endif

}  // namespace blaze
