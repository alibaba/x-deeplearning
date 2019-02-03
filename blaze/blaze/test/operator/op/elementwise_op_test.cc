/*
 * \file elementwise_op_test.cc
 * \brief The elementwise op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestElementwiseAdd, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_add_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseAdd, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_add_net_v1_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseAdd1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_add_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseAdd1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_add_net_v2_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseSub, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_sub_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseSub, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_sub_net_v1_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseSub1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_sub_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseSub1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_sub_net_v2_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseMul, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_mul_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseMul, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_mul_net_v1_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseMul1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_mul_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseMul1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_mul_net_v2_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseDiv, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_div_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseDiv, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_div_net_v1_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseDiv1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_div_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseDiv1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_div_net_v2_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseEqual, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_equal_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseEqual, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_equal_net_v1_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseEqual1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_equal_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseEqual1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_equal_net_v2_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseNotEqual, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_not_equal_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseNotEqual, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_not_equal_net_v1_gpu.conf", output_str);
}
#endif

#ifdef USE_CUDA
TEST(TestElementwiseNotEqual1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_not_equal_net_v2_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseNotEqual1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_not_equal_net_v2_cpu.conf", output_str);
}

TEST(TestElementwiseMax, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_max_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseMax, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_max_net_v1_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseMax1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_max_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseMax1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_max_net_v2_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseMin, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_min_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseMin, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_min_net_v1_gpu.conf", output_str);
}
#endif

#ifdef USE_CUDA
TEST(TestElementwiseMin1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_min_net_v2_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseMin1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_min_net_v2_cpu.conf", output_str);
}

TEST(TestElementwiseBroadcastTo, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_broadcast_to_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseBroadcastTo, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_broadcast_to_net_v1_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseBroadcastTo1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_broadcast_to_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseBroadcastTo1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_broadcast_to_net_v2_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseWhere, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_where_net_v1_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseWhere, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_where_net_v1_gpu.conf", output_str);
}
#endif

TEST(TestElementwiseWhere1, TestCPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_where_net_v2_cpu.conf", output_str);
}

#ifdef USE_CUDA
TEST(TestElementwiseWhere1, TestGPU) {
  std::vector<std::string> output_str;
  output_str.push_back("output");
  TestOperatorOutput<float>("./utest_data/operator/op/elementwise/elementwise_where_net_v2_gpu.conf", output_str);
}
#endif

}  // namespace blaze
