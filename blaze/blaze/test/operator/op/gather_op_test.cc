/*
 * \file gather_op_test.cc
 * \brief The gather op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestGather, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/gather/gather_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestGather, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/gather/gather_net_v1_gpu.conf", output_str);
}
#endif

TEST(TestGather1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/gather/gather_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestGather1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/gather/gather_net_v2_gpu.conf", output_str);
}
#endif

}  // namespace blaze
