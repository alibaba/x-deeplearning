/*
 * \file cartesian_product_op_test.cc
 * \brief The cartesian product op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

/*
TEST(TestCartesianProduct, TestCase1) {
  std::vector<std::string> output_str;
  output_str.resize(1);
  output_str[0] = "cross_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/cartesian_product/cartesian_product_v1.conf", output_str);

  output_str[0] = "cross_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/cartesian_product/cartesian_product_v1.conf", output_str);

  output_str[0] = "cross_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/cartesian_product/cartesian_product_v1.conf", output_str);
}

TEST(TestCartesianProduct, TestCase2) {
  std::vector<std::string> output_str;
  output_str.resize(1);
  output_str[0] = "cross_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/cartesian_product/cartesian_product_v2.conf", output_str);

  output_str[0] = "cross_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/cartesian_product/cartesian_product_v2.conf", output_str);

  output_str[0] = "cross_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/cartesian_product/cartesian_product_v2.conf", output_str);
}*/

TEST(TestCartesianProduct, TestCase3) {
  std::vector<std::string> output_str;
  output_str.resize(1);
  output_str[0] = "cross_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/cartesian_product/cartesian_product_v3.conf", output_str);

  output_str[0] = "cross_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/cartesian_product/cartesian_product_v3.conf", output_str);

  output_str[0] = "cross_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/cartesian_product/cartesian_product_v3.conf", output_str);
}

}  // namespace blaze