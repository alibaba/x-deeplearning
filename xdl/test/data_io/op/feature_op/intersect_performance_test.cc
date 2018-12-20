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
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "test/data_io/op/feature_op/op_test_tool.h"
#include "test/data_io/op/feature_op/performance_test_data.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_func/intersect.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_op_factory.h"

using xdl::io::FeatureLine;
using xdl::io::Feature;

TEST(IntersectPerformanceTest, Default) {
  if (testing::internal::GetArgvs().size() != 2 || testing::internal::GetArgvs()[1] != "x") {
    printf("Usage: %s x\n", testing::internal::GetArgvs()[0].c_str());
    return;
  }

  using xdl::io::ExprNode;
  using xdl::io::Intersect;
  using xdl::io::MultiFeaOpType;
  using xdl::io::MultiFeatureOp;
  using xdl::io::MultiFeatureOpFactory;

  const int feature_line_num = 100000;
  const int feature_num = 20;
  int size_range = 5;
  int key_range = 1;
  float value_range = 2.;
  int name_range = 5;

  FeatureLinePerformanceTestData test_data;

  MultiFeatureOp *multi_feature_op = MultiFeatureOpFactory::Get(MultiFeaOpType::kMergeFeatureOp,
                                                                Intersect::CombineKey,
                                                                Intersect::CombineValue);
  printf("start:\n");
  double begin_time, elapsed_time = 0.;
  for (int i = 0; i < feature_line_num; ++i) {
    std::vector<const xdl::io::Feature *> features;
    FeatureLine feature_line;
    test_data.Generate(feature_line, features, size_range, name_range,
                       key_range, value_range, feature_num);
    Feature result_feature;

    std::vector<ExprNode> nodes;
    std::vector<const ExprNode *> source_nodes;
    ExprNode result_node;
    OpTestTool::Transfer(features, result_feature, nodes, source_nodes, result_node);

    begin_time = OpTestTool::GetTime();
    multi_feature_op->Run(source_nodes, &result_node);
    elapsed_time += OpTestTool::GetTime() - begin_time;
  }
  printf("elapsed time = %g s = %g s/per\n", elapsed_time, elapsed_time / feature_line_num);

  MultiFeatureOpFactory::Release(multi_feature_op);
}
