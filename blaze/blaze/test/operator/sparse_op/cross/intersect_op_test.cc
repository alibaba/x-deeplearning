/*
 * \file intersect_op_test.cc
 * \brief The intersect op test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/test/operator/operator_test_common.h"

namespace blaze {

TEST(TestIntersect, TestCPU) {
  std::vector<std::string> output_str;
  output_str.resize(1);
  output_str[0] = "cross_ids";
  TestOperatorOutput<int64_t>("./utest_data/operator/sparse_op/cross/intersect/intersect_v1.conf", output_str);

  output_str[0] = "cross_values";
  TestOperatorOutput<float>("./utest_data/operator/sparse_op/cross/intersect/intersect_v1.conf", output_str);

  output_str[0] = "cross_fea_num";
  TestOperatorOutput<int32_t>("./utest_data/operator/sparse_op/cross/intersect/intersect_v1.conf", output_str);
}

}  // namespace blaze