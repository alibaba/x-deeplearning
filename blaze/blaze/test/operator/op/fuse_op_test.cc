/*
 * \file fuse_op_test.cc
 * \brief The fuse op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestLegacyFuse, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/fuse/fuse_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestLegacyFuse, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/fuse/fuse_net_v1_gpu.conf", output_str);
}
#endif

TEST(TestLegacyFuse1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/fuse/fuse_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestLegacyFuse1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/fuse/fuse_net_v2_gpu.conf", output_str);
}
#endif

}  // namespace blaze
