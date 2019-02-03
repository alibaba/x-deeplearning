/*
 * \file fused_slice_concat_op_test.cc
 * \brief The fused slice concat test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestFusedSliceConcat, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_slice_concat/fused_slice_concat_net_v1_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_slice_concat/fused_slice_concat_net_v2_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_slice_concat/fused_slice_concat_net_v3_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestFusedSliceConcat, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_slice_concat/fused_slice_concat_net_v1_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_slice_concat/fused_slice_concat_net_v2_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_slice_concat/fused_slice_concat_net_v3_gpu.conf", output_str);
}
#endif

}  // namespace blaze
