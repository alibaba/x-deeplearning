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

#include <thread>
#include <vector>

#include "test/data_io/op/feature_op/op_test_tool.h"
#include "xdl/data_io/op/feature_op/expr/internal_feature.h"
#include "xdl/proto/sample.pb.h"

using xdl::io::Feature;
using xdl::io::FeatureValue;
using xdl::io::InternalValue;

TEST(PbPerformanceTest, Default) {
  if (testing::internal::GetArgvs().size() != 2 || testing::internal::GetArgvs()[1] != "x") {
    printf("Usage: %s x\n", testing::internal::GetArgvs()[0].c_str());
    return;
  }

  const int size = 1000000;

  char *s = nullptr;
  Feature feature;
  std::vector<InternalValue> vec(size);
  std::vector<InternalValue *> vecs(size);
  for (int i = 0; i < size; ++i) {
    FeatureValue *feature_value = feature.add_values();
    feature_value->set_key(i);
    feature_value->set_value(i * 1.1);
    const InternalValue internal_value = {i, i * 1.1F};
    vec[i] = std::move(internal_value);
    vecs[i] = new InternalValue();
    vecs[i]->key_ = i;
    vecs[i]->value_ = i * 1.1F;
    s = new char[100 * 1000 * 1000];
  }

  double time0 = OpTestTool::GetTime();
  for (int i = 0; i < size; ++i) {
    FeatureValue *feature_value = feature.mutable_values(i);
    feature_value->set_key(i);
  }
  printf("pb: %g\n", OpTestTool::GetTime() - time0);

  double time1 = OpTestTool::GetTime();
  for (int i = 0; i < size; ++i) {
    InternalValue &internal_value = vec[i];
    internal_value.key_ = i;
  }
  printf("vec: %g\n", OpTestTool::GetTime() - time1);  // 比遍历pb快45倍

  double time2 = OpTestTool::GetTime();
  for (int i = 0; i < size; ++i) {
    InternalValue *internal_value = vecs[i];
    internal_value->key_ = i;
  }
  printf("vecs: %g\n", OpTestTool::GetTime() - time2);

  if (s)  delete s;
}
