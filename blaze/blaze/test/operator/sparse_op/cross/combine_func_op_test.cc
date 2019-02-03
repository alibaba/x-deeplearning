/*
 * \file combine_func_op_test.cc
 * \brief The combine func op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestCombineFunc, TestFuncID) {
  std::vector<std::string> output_str;
  output_str.resize(1);
  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_id_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_id_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_id_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncLogID) {
  std::vector<std::string> output_str;
  output_str.resize(1);
  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_id_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_id_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_id_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncSum) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_sum_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_sum_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_sum_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncLogSum) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_sum_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_sum_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_sum_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncMax) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_max_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_max_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_max_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncLogMax) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_max_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_max_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_max_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncMin) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_min_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_min_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_min_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncLogMin) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_min_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_min_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_min_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncAvg) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_avg_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_avg_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_avg_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncLogAvg) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_avg_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_avg_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_avg_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncCos) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_cos_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_cos_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_cos_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncLogCos) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_cos_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_cos_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_cos_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncDotSum) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_dot_sum_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_dot_sum_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_dot_sum_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncLogDotSum) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_dot_sum_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_dot_sum_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_log_dot_sum_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncDotL1Norm) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_dot_l1_norm_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_dot_l1_norm_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_dot_l1_norm_v1.conf", output_str);
}

TEST(TestCombineFunc, TestFuncDotL2Norm) {
  std::vector<std::string> output_str;
  output_str.resize(1);

  output_str[0] = "output_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_dot_l2_norm_v1.conf", output_str);

  output_str[0] = "output_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_dot_l2_norm_v1.conf", output_str);

  output_str[0] = "output_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/combine_func/combine_func_dot_l2_norm_v1.conf", output_str);
}

}  // namespace blaze