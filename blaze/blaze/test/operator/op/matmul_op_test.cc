/*
 * \file matmul_op_test.cc
 * \brief The matmul op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestMatMul, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/matmul/matmul_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestMatMul, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/matmul/matmul_net_v1_gpu.conf", output_str);
}
#endif

}  // namespace blaze
