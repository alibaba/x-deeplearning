/*
 * \file softmax_op_test.cc
 * \brief The softmax op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestTanh, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/tanh/tanh_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestTanh, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/tanh/tanh_net_v1_gpu.conf", output_str);
  // Test fp16
  TestOperatorOutput<float16>("./utest_data/operator/op/tanh/tanh_net_v1_fp16_gpu.conf", output_str);
}
#endif

}  // namespace blaze
