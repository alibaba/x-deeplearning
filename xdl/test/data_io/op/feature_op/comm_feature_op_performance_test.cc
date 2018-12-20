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

#include <string>
#include <unordered_map>

#include "test/data_io/op/feature_op/op_test_tool.h"
#include "test/data_io/op/feature_op/performance_test_data.h"
#include "xdl/data_io/op/feature_op/feature_op.h"

TEST(CommFeatureOpPerformanceTest, Default) {
  if (testing::internal::GetArgvs().size() != 2 || testing::internal::GetArgvs()[1] != "x") {
    printf("Usage: %s x\n", testing::internal::GetArgvs()[0].c_str());
    return;
  }

  using xdl::io::FeatureNameVec;
  using xdl::io::FeatureOP;
  using xdl::io::SampleGroup;
  using xdl::io::FeatureTable;
  using xdl::io::FeatureLine;

  int feature_line_num = 20000;
  int key_range = 1000;
  float value_range = 2.;

  FeatureLinePerformanceTestData test_data;
  SampleGroup sample_group;

  std::vector<int> sizes;
  const int feature_num = 100;
  for (int i = 0; i< feature_num; ++i) {
    sizes.push_back(OpTestTool::Rand(3));
  }
  sizes['W' - 'A'] = 1;
  FeatureTable *feature_table = sample_group.add_feature_tables();
  for (int i = 0; i < feature_line_num; ++i) {
    std::vector<const xdl::io::Feature *> features;
    FeatureLine *feature_line = feature_table->add_feature_lines();
    test_data.GenerateFixed(*feature_line, features, sizes,
                            key_range, value_range, feature_num);
    feature_line->set_refer(0);
  }

  std::vector<int> comm_sizes;
  const int comm_feature_num = 100;
  for (int i = 0; i< comm_feature_num; ++i) {
    comm_sizes.push_back(OpTestTool::Rand(3));
  }
  comm_sizes['z' - 'a'] = 1000;
  FeatureTable *comm_feature_table = sample_group.add_feature_tables();
  for (int i = 0; i < 1; ++i) {
    std::vector<const xdl::io::Feature *> features;
    FeatureLine *feature_line = comm_feature_table->add_feature_lines();;
    test_data.GenerateFixed(*feature_line, features, comm_sizes,
                            key_range, value_range, comm_feature_num, 'a');
  }

  std::vector<FeatureNameVec> feature_name_vecs(2);
  feature_name_vecs[0].push_back("AAAAAAW");
  feature_name_vecs[1].push_back("aaaaaaz");
  std::vector<std::string> conf_lines = {
    "Name=z_ad_cate_pv_14_match_sum; Expr=sum(match(log(AAAAAAW),aaaaaaz)); Type=Numeric;"
  };

  FeatureOP feature_op;
  feature_op.Init(conf_lines, feature_name_vecs);
  feature_op.Run(&sample_group);

  const int sg_num = 50;
  double begin_time, elapsed_time = 0.;
  for (int k = 0; k < sg_num; ++k) {
    begin_time = OpTestTool::GetTime();
    feature_op.Run(&sample_group);
    elapsed_time += OpTestTool::GetTime() - begin_time;
    const FeatureTable &feature_table = sample_group.feature_tables(0);
    for (int i = 0; i < feature_table.feature_lines_size(); ++i) {
      const FeatureLine &feature_line = feature_table.feature_lines(i);
      EXPECT_EQ(feature_line.features_size(), feature_num + 2 + k);
      EXPECT_EQ(feature_line.features(feature_line.features_size() - 1).name(), "z_ad_cate_pv_14_match_sum");
    }
  }
  printf("elapsed time = %g s = %g s/per sg\n", elapsed_time, elapsed_time / sg_num);
}
