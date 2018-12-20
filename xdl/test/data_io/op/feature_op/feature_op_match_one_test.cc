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
#include <thread>
#include <unordered_map>

#include "test/data_io/op/feature_op/test_data.h"
#include "xdl/data_io/op/feature_op/feature_op.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_func/log_feature.h"

using xdl::io::FeatureNameVec;
using xdl::io::FeatureOP;
using xdl::io::SampleGroup;
using xdl::io::FeatureTable;
using xdl::io::FeatureLine;
using xdl::io::LogFeature;

void SchemaAddOption(xdl::io::Schema &schema, const char *name, const char *dsl = nullptr) {
  xdl::io::FeatureOption *feature_option = new xdl::io::FeatureOption();
  feature_option->set_type(xdl::io::kSparse);
  feature_option->set_table(0);
  feature_option->set_name(name);
  if (dsl != nullptr)  feature_option->set_dsl(dsl);
  schema.Add(feature_option);
}

TEST(FeatureOpTest, MatchOne) {
  bool is_sort = false;
  const int num = 3;
  std::string name_arr[] = {"ad_cate_id", "fake_id", "nick_cate_pv_14"};
  int size_arr[] = {3, 5, 4};
  std::vector<int64_t> key_arr[num] = { std::vector<int64_t>({1, 2, 5}),
                                        std::vector<int64_t>({1, 2, 3, 4, 5}),
                                        std::vector<int64_t>({0, 1, 2, 3}) };
  std::vector<float> value_arr[num] = { std::vector<float>({0.2, 0.1, 0.3}),
                                        std::vector<float>({0.4, 0.1, 0.2, 0.5, 0.3}),
                                        std::vector<float>({0.1, 0.2, 3e38, 0.4}) };

  std::vector<const Feature *> features;
  SampleGroup sample_group;
  FeatureTable *feature_table = sample_group.add_feature_tables();
  FeatureLine *feature_line = feature_table->add_feature_lines();
  FeatureLineTestData test_data;
  test_data.Generate(*feature_line, features, name_arr, size_arr, key_arr, value_arr, num, is_sort, false);

  FeatureOP feature_op;
  xdl::io::Schema schema;
  SchemaAddOption(schema, "ad_cate_id");
  SchemaAddOption(schema, "nick_cate_pv_14");
  SchemaAddOption(schema, "z_ad_cate_pv_14_match_sum",
                  "Name=z_ad_cate_pv_14_match_sum; Expr=logavg(avg(logmin(logmax(logsum(match(min(ad_cate_id),nick_cate_pv_14)))))); Type=Numeric;");
  // Expr = sum(match(2:0.1 ~ 2:log(0.3))) = 2:0.1*3e38
  feature_op.set_schema(&schema);
  feature_op.Init(std::map<std::string, std::string>());
  feature_op.Run(&sample_group);

  const FeatureLine &result_feature_line = sample_group.feature_tables(0).feature_lines(0);
  EXPECT_EQ(result_feature_line.features_size(), 4);
  // check origin test data
  for (int i = 0; i < num; ++i) {
    const Feature &feature = result_feature_line.features(i);
    EXPECT_EQ(feature.name(), name_arr[i]);
    EXPECT_EQ(feature.values_size(), size_arr[i]);
    for (int j = 0; j < feature.values_size(); ++j) {
      test_data.ExpectKeyEq(feature.values(j).key(), key_arr[i][j], j);
      test_data.ExpectValueNear(feature.values(j).value(), value_arr[i][j], j);
    }
  }
  // check result
  const Feature &feature = result_feature_line.features(3);
  EXPECT_EQ(feature.name(), "z_ad_cate_pv_14_match_sum");
  EXPECT_EQ(feature.values_size(), 1);
  EXPECT_EQ(feature.values(0).key(), 2);
  float expected_result = 0.1 * 3e38;
  for (int i = 0; i < 4; ++i)  expected_result = LogFeature::TransformValue(expected_result);
  test_data.ExpectValueNear(feature.values(0).value(), expected_result, 0);
}
