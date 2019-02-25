/*
 * \file fused_parallel_gemm_op_test.cc
 * \brief The fused parallel gemm test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestFusedParallelGemm, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_parallel_gemm/fused_parallel_gemm_net_v1_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_parallel_gemm/fused_parallel_gemm_net_v2_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_parallel_gemm/fused_parallel_gemm_net_v3_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_parallel_gemm/fused_parallel_gemm_net_v4_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_parallel_gemm/fused_parallel_gemm_net_v5_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestFusedParallelGemm, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_parallel_gemm/fused_parallel_gemm_net_v1_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_parallel_gemm/fused_parallel_gemm_net_v2_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_parallel_gemm/fused_parallel_gemm_net_v3_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_parallel_gemm/fused_parallel_gemm_net_v4_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/fused_op/fused_parallel_gemm/fused_parallel_gemm_net_v5_gpu.conf", output_str);
}
#endif

}  // namespace blaze
