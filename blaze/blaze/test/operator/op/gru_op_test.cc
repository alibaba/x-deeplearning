/*
 * \file gru_op_test.cc
 * \brief The gru op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestGRU, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_1_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_2_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_1_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_2_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_4_5_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_4_20_9_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_7_15_23_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_108_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_108_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_3_108_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_50_108_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_50_108_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_30_128_cpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_1_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_2_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_1_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_2_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_4_5_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_4_20_9_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_7_15_23_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_108_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_108_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_3_108_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_50_108_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_50_108_cpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_30_128_cpu_onnx.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestGRU, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_1_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_2_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_1_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_2_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_4_5_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_4_20_9_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_7_15_23_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_108_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_108_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_3_108_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_50_108_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_50_108_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_30_128_gpu.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_1_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_2_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_1_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_2_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_4_5_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_4_20_9_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_7_15_23_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_1_108_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_2_108_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_3_108_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_1_50_108_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_50_108_gpu_onnx.conf", output_str);
  TestOperatorOutput<float>("./utest_data/operator/op/gru/gru_test_2_30_128_gpu_onnx.conf", output_str);
  
  TestOperatorOutput<float16, false>("./utest_data/operator/op/gru/gru_test_1_2_2_fp16_gpu.conf", output_str);
}
#endif

}  // namespace blaze
