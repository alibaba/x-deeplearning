/*
 * \file split_op_test.cc
 * \brief The split op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestSlice, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output0");
  output_str.push_back("output1");
  TestOperatorOutput<float>("./utest_data/operator/op/split/split_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestSlice, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output0");
  output_str.push_back("output1");
  TestOperatorOutput<float>("./utest_data/operator/op/split/split_net_v1_gpu.conf", output_str);
}
#endif

TEST(TestSlice1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output0");
  output_str.push_back("output1");
  TestOperatorOutput<float>("./utest_data/operator/op/split/split_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestSlice1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output0");
  output_str.push_back("output1");
  TestOperatorOutput<float>("./utest_data/operator/op/split/split_net_v2_gpu.conf", output_str);
}
#endif

}  // namespace blaze
