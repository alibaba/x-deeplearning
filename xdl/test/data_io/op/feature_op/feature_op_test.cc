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

TEST(FeatureOpTest, Default) {
  bool is_sort = false;
  const int num = 3;
  std::string name_arr[] = {"ad_cate_id", "fake_id", "nick_cate_pv_14"};
  int size_arr[] = {3, 5, 4};
  std::vector<int64_t> key_arr[num] = { std::vector<int64_t>({2, 0, 5}),
                                        std::vector<int64_t>({3, 1, 4, 5, 2}),
                                        std::vector<int64_t>({0, 1, 2, 3}) };
  std::vector<float> value_arr[num] = { std::vector<float>({0.1, 0.2, 0.3}),
                                        std::vector<float>({0.4, 0.1, 0.2, 0.5, 0.3}),
                                        std::vector<float>({0.1, 0.2, 0.3, 0.4}) };

  std::vector<const Feature *> features;
  SampleGroup sample_group;
  FeatureTable *feature_table = sample_group.add_feature_tables();
  FeatureLine *feature_line = feature_table->add_feature_lines();
  FeatureLineTestData test_data;
  test_data.Generate(*feature_line, features, name_arr, size_arr, key_arr, value_arr, num, is_sort, false);

  std::vector<FeatureNameVec> feature_name_vecs(1);
  feature_name_vecs[0].push_back("ad_cate_id");
  feature_name_vecs[0].push_back("nick_cate_pv_14");
  std::vector<std::string> dsl_arr = {
    "Name=ad_cate_pv_14_match_sum; Expr=min(max(sum(match(ad_cate_id,log(nick_cate_pv_14))))); Type=Numeric;"
  };  // Expr = sum(match(0:0.2, 2:0.1 ~ 0:log(0.1), 2:log(0.3))) = 0.2*log(0.1) + 0.1*log(0.3)
  FeatureOP feature_op;
  feature_op.Init(dsl_arr, feature_name_vecs);
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
  EXPECT_EQ(feature.name(), "ad_cate_pv_14_match_sum");
  EXPECT_EQ(feature.values_size(), 1);
  EXPECT_FALSE(feature.values(0).has_key());
  float expected_result = 0.2 * LogFeature::TransformValue(0.1) + 0.1 * LogFeature::TransformValue(0.3);
  test_data.ExpectValueNear(feature.values(0).value(), expected_result, 0);
}

class RunThread {
 public:
  void Init(int rank, FeatureOP *feature_op, std::string name_arr[], int size_arr[], bool is_sort) {
    rank_ = rank;
    feature_op_ = feature_op;
    name_arr_ = name_arr;
    size_arr_ = size_arr;
    for (int i = 0; i < num_; ++i) {
      for (size_t j = 0; j < value_arr_[i].size(); ++j) {
        value_arr_[i][j] = ChangeValue(value_arr_[i][j]);
      }
    }

    std::vector<const Feature *> features;
    FeatureTable *feature_table = sample_group_.add_feature_tables();
    FeatureLine *feature_line = feature_table->add_feature_lines();
    test_data_.Generate(*feature_line, features, name_arr, size_arr, key_arr_, value_arr_, num_, is_sort, false);
  }

  void Run() {
    feature_op_->Run(&sample_group_);
  }

  void Check() {
    const FeatureLine &result_feature_line = sample_group_.feature_tables(0).feature_lines(0);
    EXPECT_EQ(result_feature_line.features_size(), 4);
    // check origin test data
    for (int i = 0; i < num_; ++i) {
      const Feature &feature = result_feature_line.features(i);
      EXPECT_EQ(feature.name(), name_arr_[i]);
      EXPECT_EQ(feature.values_size(), size_arr_[i]);
      for (int j = 0; j < feature.values_size(); ++j) {
        test_data_.ExpectKeyEq(feature.values(j).key(), key_arr_[i][j], j);
        test_data_.ExpectValueNear(feature.values(j).value(), value_arr_[i][j], j);
      }
    }
    // check result
    const Feature &feature = result_feature_line.features(3);
    EXPECT_EQ(feature.name(), "ad_cate_pv_14_match_sum");
    EXPECT_EQ(feature.values_size(), 1);
    EXPECT_FALSE(feature.values(0).has_key());
    float expected_result = ChangeValue(0.2) * LogFeature::TransformValue(ChangeValue(0.1))
                            + ChangeValue(0.1) * LogFeature::TransformValue(ChangeValue(0.3));
    test_data_.ExpectValueNear(feature.values(0).value(), expected_result, 0);
  }

  inline float ChangeValue(float value) const {
    return value + 0.1 * rank_;
  }

 private:
  int rank_;
  FeatureOP *feature_op_;
  SampleGroup sample_group_;
  FeatureLineTestData test_data_;
  const std::string *name_arr_;
  const int *size_arr_;

  static const int num_ = 3;
  std::vector<int64_t> key_arr_[num_] = { std::vector<int64_t>({2, 0, 5}),
                                          std::vector<int64_t>({3, 1, 4, 5, 2}),
                                          std::vector<int64_t>({0, 1, 2, 3}) };
  std::vector<float> value_arr_[num_] = { std::vector<float>({0.1, 0.2, 0.3}),
                                          std::vector<float>({0.4, 0.1, 0.2, 0.5, 0.3}),
                                          std::vector<float>({0.1, 0.2, 0.3, 0.4}) };
};

TEST(FeatureOpTest, MultiThread) {
  bool is_sort = false;
  std::string name_arr[] = {"ad_cate_id", "fake_id", "nick_cate_pv_14"};
  int size_arr[] = {3, 5, 4};

  std::vector<FeatureNameVec> feature_name_vecs(1);
  feature_name_vecs[0].push_back("ad_cate_id");
  feature_name_vecs[0].push_back("nick_cate_pv_14");
  std::vector<std::string> dsl_arr = {
    "Name=ad_cate_pv_14_match_sum; Expr=sum(match(ad_cate_id,log(nick_cate_pv_14))); Type=Numeric;"
  };  // Expr = sum(match(0:0.2, 2:0.1 ~ 0:log(0.1), 2:log(0.3))) = 0.2*log(0.1) + 0.1*log(0.3)
  FeatureOP feature_op;
  feature_op.Init(dsl_arr, feature_name_vecs);

  const int thread_num = 30;
  RunThread r[thread_num];
  std::thread *run_threads[thread_num];
  for (int t = 0; t < thread_num; ++t) {
    r[t].Init(t, &feature_op, name_arr, size_arr, is_sort);
    run_threads[t] = new std::thread(&RunThread::Run, &r[t]);
  }
  for (int t = 0; t < thread_num; ++t) {
    run_threads[t]->join();
    r[t].Check();
    delete run_threads[t];
  }
}
