/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "gtest/gtest.h"

#include <math.h>
#include <stdlib.h>
#include <vector>

#include "test/data_io/op/feature_op/op_test_tool.h"
#include "test/data_io/op/feature_op/test_data.h"
#include "xdl/data_io/op/feature_op/expr/expr_node.h"
#include "xdl/data_io/op/feature_op/feature_op_constant.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_func/dot_product.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_op_factory.h"

class FeatureLineDotProductTestData : public FeatureLineTestData {
 public:
  void Generate(xdl::io::FeatureLine &feature_line,
                std::vector<const xdl::io::Feature *> &features,
                std::string name_arr[], int size_arr[],
                std::vector<int64_t> key_arr[], std::vector<float> value_arr[],
                int feature_num, bool is_sort = false, bool is_check = true) override {
    expected_value_size_ = 1;
    FeatureLineTestData::Generate(feature_line, features,
                                  name_arr, size_arr,
                                  key_arr, value_arr,
                                  feature_num, is_sort, is_check);
  }

  void Check(xdl::io::Feature &result_feature) const override {
    EXPECT_EQ(result_feature.values_size(), expected_value_size_);
    FeatureLineTestData::Check(result_feature);
  }

 protected:
  void Combine(const FeatureValue &feature_value, size_t m,
               std::vector<int64_t> &tmp_key_arr, std::vector<float> &tmp_value_arr) override {
    using xdl::io::DotProduct;
    if (expected_key_arr_[m] == feature_value.key()) {
      aa_ += DotProduct::CombineValue(expected_value_arr_[m], expected_value_arr_[m]);
      bb_ += DotProduct::CombineValue(feature_value.value(), feature_value.value());
      ab_ += DotProduct::CombineValue(expected_value_arr_[m], feature_value.value());
      if (m == 3) {
        tmp_key_arr.push_back(0);  // xdl::io::kDefaultKey
        tmp_value_arr.push_back(ab_ / (sqrtf(aa_) + sqrtf(bb_) + xdl::io::kTraceFloat));
      }
    }
  }

 private:
  int expected_value_size_ = 1;
  float aa_ = 0.F;
  float bb_ = 0.F;
  float ab_ = 0.F;
};

TEST(DotProductTest, Default) {
  using xdl::io::ExprNode;
  using xdl::io::FeaOpType;
  using xdl::io::DotProduct;
  using xdl::io::MultiFeaOpType;
  using xdl::io::MultiFeatureOp;
  using xdl::io::MultiFeatureOpFactory;

  const int num = 2;
  std::string name_arr[] = {"0", "1"};
  int size_arr[] = {4, 4};
  std::vector<int64_t> key_arr[num] = { std::vector<int64_t>({0, 1, 2, 3}),
                                        std::vector<int64_t>({0, 1, 2, 3}) };
  std::vector<float> value_arr[num] = { std::vector<float>({0.1, 0.2, 0.3, 0.4}),
                                        std::vector<float>({0.4, 0.1, 0.2, 0.5}) };

  std::vector<const Feature *> features;
  FeatureLine feature_line;
  Feature result_feature;
  FeatureLineDotProductTestData test_data;
  test_data.Generate(feature_line, features, name_arr, size_arr, key_arr, value_arr, num);

  std::vector<ExprNode> nodes;
  std::vector<const ExprNode *> source_nodes;
  ExprNode result_node;
  OpTestTool::Transfer(features, result_feature, nodes, source_nodes, result_node);

  MultiFeatureOp *multi_feature_op = MultiFeatureOpFactory::Get(MultiFeaOpType::kVectorFeatureOp,
                                                                nullptr,
                                                                DotProduct::CombineValue);
  multi_feature_op->Run(source_nodes, &result_node);

  test_data.Check(result_feature);

  MultiFeatureOpFactory::Release(multi_feature_op);
}
