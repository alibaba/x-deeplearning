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
#include "xdl/core/framework/tensor.h"
#include "xdl/core/utils/file_utils.h"
#include "xdl/core/backend/tf/tf_runner.h"
#include "xdl/core/backend/tf/convert_utils.h"

using namespace xdl;

static void TensorEqual(const tensorflow::Tensor& src, 
                        const std::vector<float>& dst) {
  ASSERT_EQ(src.NumElements(), dst.size());
  const float* src_ptr = src.flat<float>().data();
  for (size_t i = 0; i < dst.size(); ++i) {
    ASSERT_FLOAT_EQ(src_ptr[i], dst[i]);
  }
}

TEST(TFBackendTest, RunnerTest) {
  TFRunner runner;
  std::string graph_path = "../test_data/backend/tf_test.pb";
  std::string graph_def = FileUtils::ReadLocalBinaryFile(graph_path);
  ASSERT_TRUE(!graph_def.empty());
  ASSERT_TRUE(Status::Ok() == runner.Init(graph_def));
  TFRunner::InputList inputs;
  tensorflow::Tensor a = MakeTensor<float>({1,1}, {2});
  tensorflow::Tensor b = MakeTensor<float>({2,2}, {2});
  tensorflow::Tensor y = MakeTensor<float>({2,3}, {2});
  inputs.push_back({"a:0", a});
  inputs.push_back({"b:0", b});
  inputs.push_back({"y:0", y});

  std::vector<std::string> op_names;
  op_names.push_back("Add:0");
  op_names.push_back("div:0");
  op_names.push_back("Identity:0");
  op_names.push_back("Identity_1:0");
  op_names.push_back("Identity_2:0");
  
  std::vector<tensorflow::Tensor> results;
  ASSERT_TRUE(Status::Ok() == 
              runner.Run(inputs, op_names, &results));

  TensorEqual(results[0], {3,3});
  TensorEqual(results[1], {-0.5,0});
  TensorEqual(results[2], {-0.5,-0.5});
  TensorEqual(results[3], {-0.5,-0.5});
  TensorEqual(results[4], {0.5,0.5});
}

