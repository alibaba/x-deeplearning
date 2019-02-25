/*
 * \file constant_fill_op_test.cc
 * \brief The constant fill op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestConstantFill, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/constant_fill/constant_fill_net_v1_cpu.conf", output_str);
  TestOperatorOutput<float16>("./utest_data/operator/op/constant_fill/constant_fill_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestConstantFill, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/constant_fill/constant_fill_net_v1_gpu.conf", output_str);
  TestOperatorOutput<float16>("./utest_data/operator/op/constant_fill/constant_fill_net_v2_gpu.conf", output_str);
  TestOperatorOutput<int32_t>("./utest_data/operator/op/constant_fill/constant_fill_net_v3_gpu.conf", output_str);
}
#endif

}  // namespace blaze
