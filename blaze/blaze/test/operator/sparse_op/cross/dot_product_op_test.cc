/*
 * \file dot_product_product_op_test.cc
 * \brief The dot product op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestDotProduct, TestCPU) {
  std::vector<std::string> output_str;
  output_str.resize(1);
  output_str[0] = "z_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/dot_product/dot_product_v1.conf", output_str);

  output_str[0] = "z_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/dot_product/dot_product_v1.conf", output_str);

  output_str[0] = "z_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/dot_product/dot_product_v1.conf", output_str);
}

}  // namespace blaze