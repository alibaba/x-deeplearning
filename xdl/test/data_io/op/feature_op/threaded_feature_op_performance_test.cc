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
#include <vector>

#include "test/data_io/op/feature_op/op_test_tool.h"
#include "test/data_io/op/feature_op/performance_test_data.h"
#include "xdl/data_io/op/feature_op/feature_op.h"

using xdl::io::FeatureNameVec;
using xdl::io::FeatureOP;
using xdl::io::FeatureTable;
using xdl::io::FeatureLine;
using xdl::io::SampleGroup;

class CommRunThread {
 public:
  void Init(int rank, FeatureOP *feature_op, int feature_line_num, int key_range, float value_range) {
    feature_op_ = feature_op;

    std::vector<int> sizes;
    const int feature_num = 100;
    for (int i = 0; i< feature_num; ++i) {
      sizes.push_back(OpTestTool::Rand(2));
    }
    sizes['W' - 'A'] = 1;
    FeatureTable *feature_table = sample_group_.add_feature_tables();
    for (int i = 0; i < feature_line_num; ++i) {
      std::vector<const xdl::io::Feature *> features;
      FeatureLine *feature_line = feature_table->add_feature_lines();
      test_data_.GenerateFixed(*feature_line, features, sizes,
                               key_range, value_range, feature_num);
      feature_line->set_refer(0);
    }

    std::vector<int> comm_sizes;
    const int comm_feature_num = 100;
    for (int i = 0; i< comm_feature_num; ++i) {
      comm_sizes.push_back(OpTestTool::Rand(3));
    }
    comm_sizes['z' - 'a'] = 1000;
    FeatureTable *comm_feature_table = sample_group_.add_feature_tables();
    for (int i = 0; i < 1; ++i) {
      std::vector<const xdl::io::Feature *> features;
      FeatureLine *feature_line = comm_feature_table->add_feature_lines();;
      test_data_.GenerateFixed(*feature_line, features, comm_sizes,
                               key_range, value_range, comm_feature_num, 'a');
    }
    feature_op_->Run(&sample_group_);
  }

  void Run(int sg_num) {
    double begin_time = OpTestTool::GetTime();
    for (int k = 0; k < sg_num; ++k)  feature_op_->Run(&sample_group_);
    elapsed_time_ += (OpTestTool::GetTime() - begin_time);
  }

  double elapsed_time() const { return elapsed_time_; }

 private:
  FeatureOP *feature_op_;
  SampleGroup sample_group_;
  FeatureLinePerformanceTestData test_data_;
  double elapsed_time_ = 0.;
};

TEST(ThreadedFeatureOpPerformanceTest, Default) {
  if (testing::internal::GetArgvs().size() != 2 || testing::internal::GetArgvs()[1] != "x") {
    printf("Usage: %s x\n", testing::internal::GetArgvs()[0].c_str());
    return;
  }

  const int sg_num = 50;
  int feature_line_num = 20000;
  int key_range = 1000;
  float value_range = 2.;

  std::vector<FeatureNameVec> feature_name_vecs(2);
  feature_name_vecs[0].push_back("AAAAAAW");
  feature_name_vecs[1].push_back("aaaaaaz");
  std::vector<std::string> conf_lines = {
    "Name=z_ad_cate_pv_14_match_sum; Expr=sum(value(match(log(AAAAAAW),aaaaaaz))); Type=Numeric;"
  };
  FeatureOP feature_op;
  feature_op.Init(conf_lines, feature_name_vecs);

  const int thread_num = 10;
  CommRunThread r[thread_num];
  std::thread *run_threads[thread_num];
  for (int t = 0; t < thread_num; ++t) {
    r[t].Init(t, &feature_op, feature_line_num, key_range, value_range);
  }

  printf("Start test.\n");
  double statis_time = 0.;
  double begin_time = OpTestTool::GetTime();
  for (int t = 0; t < thread_num; ++t) {
    run_threads[t] = new std::thread(&CommRunThread::Run, &r[t], sg_num);
  }
  for (int t = 0; t < thread_num; ++t) {
    run_threads[t]->join();
    statis_time += r[t].elapsed_time();
  }
  double elapsed_time = (OpTestTool::GetTime() - begin_time);
  printf("elapsed time = %g s = %g s/per sg\n", elapsed_time, elapsed_time / sg_num / thread_num);
  printf("statis time = %g s = %g s/per sg\n", statis_time, statis_time / sg_num / thread_num / thread_num);

  for (int t = 0; t < thread_num; ++t) {
    delete run_threads[t];
  }
}
