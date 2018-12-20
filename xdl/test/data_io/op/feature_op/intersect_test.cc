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
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_func/intersect.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_op_factory.h"

class FeatureLineIntersectTestData : public FeatureLineTestData {
 protected:
  void Combine(const FeatureValue &feature_value, size_t m,
               std::vector<int64_t> &tmp_key_arr, std::vector<float> &tmp_value_arr) override {
    using xdl::io::Intersect;
    if (expected_key_arr_[m] == feature_value.key()) {
      tmp_key_arr.push_back(Intersect::CombineKey(expected_key_arr_[m], feature_value.key()));
      tmp_value_arr.push_back(Intersect::CombineValue(expected_value_arr_[m], feature_value.value()));
    }
  }
};

TEST(IntersectTest, Default) {
  using xdl::io::ExprNode;
  using xdl::io::FeaOpType;
  using xdl::io::Intersect;
  using xdl::io::MultiFeaOpType;
  using xdl::io::MultiFeatureOp;
  using xdl::io::MultiFeatureOpFactory;

  const int num = 3;
  std::string name_arr[] = {"2", "0", "1"};
  int size_arr[] = {4, 5, 3};
  std::vector<int64_t> key_arr[num] = { std::vector<int64_t>({0, 1, 2, 3}),
                                        std::vector<int64_t>({3, 1, 4, 5, 2}),
                                        std::vector<int64_t>({2, 0, 5}) };
  std::vector<float> value_arr[num] = { std::vector<float>({0.1, 0.2, 0.3, 0.4}),
                                        std::vector<float>({0.4, 0.1, 0.2, 0.5, 0.3}),
                                        std::vector<float>({0.1, 0.2, 0.3}) };

  std::vector<const Feature *> features;
  FeatureLine feature_line;
  Feature result_feature;
  FeatureLineIntersectTestData test_data;
  test_data.Generate(feature_line, features, name_arr, size_arr, key_arr, value_arr, num);

  std::vector<ExprNode> nodes;
  std::vector<const ExprNode *> source_nodes;
  ExprNode result_node;
  OpTestTool::Transfer(features, result_feature, nodes, source_nodes, result_node);

  MultiFeatureOp *multi_feature_op = MultiFeatureOpFactory::Get(MultiFeaOpType::kMergeFeatureOp,
                                                                Intersect::CombineKey,
                                                                Intersect::CombineValue);
  multi_feature_op->Run(source_nodes, &result_node);

  test_data.Check(result_feature);

  MultiFeatureOpFactory::Release(multi_feature_op);
}
