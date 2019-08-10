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
#include "xdl/core/backend/device_singleton.h"
#include "xdl/core/backend/mxnet/mxnet_runner.h"
#include "xdl/core/backend/mxnet/convert_utils.h"

#include <stdlib.h>

using namespace xdl;

static void ArrayEqual(const mxnet::cpp::NDArray& src,
                       const std::vector<float>& dst) {
  ASSERT_EQ(src.Size(), dst.size());
  const float* src_ptr = src.GetData();
  for (size_t i = 0; i < dst.size(); ++i) {
    ASSERT_FLOAT_EQ(src_ptr[i], dst[i]);
  }
}

TEST(MxnetBackendTest, RunnerTest) {
  MxnetRunner runner;
  std::string symbol_path = "../test_data/backend/mx_test.json";
  std::string graph_def = FileUtils::ReadLocalFile(symbol_path);
  ASSERT_TRUE(!graph_def.empty());
  ASSERT_TRUE(Status::Ok() == runner.Init(graph_def, "cpu"));

  for (size_t n = 0; n < 3; ++n) {
    size_t bs = 2 + rand() / (RAND_MAX / 8);
    printf("==== bs=%lu ====\n", bs);

    Tensor a(DeviceSingleton::CpuInstance(), TensorShape({bs}), DataType::kFloat);
    Tensor b(DeviceSingleton::CpuInstance(), TensorShape({bs}), DataType::kFloat);
    Tensor y(DeviceSingleton::CpuInstance(), TensorShape({bs}), DataType::kFloat);
    mx_float* a_ptr = new mx_float[bs];
    mx_float* b_ptr = new mx_float[bs];
    mx_float* y_ptr = new mx_float[bs];
    for (int i = 0; i < bs; ++i) {
      a_ptr[i] = 1;
      b_ptr[i] = 2;
      y_ptr[i] = 2 + i;
    }
    memcpy(a.Raw<float>(), a_ptr, bs * sizeof(float));
    memcpy(b.Raw<float>(), b_ptr, bs * sizeof(float));
    memcpy(y.Raw<float>(), y_ptr, bs * sizeof(float));

    MxnetRunner::InputList inputs;
    inputs.push_back({"a", a});
    inputs.push_back({"b", b});
    inputs.push_back({"y", y});

    MxnetRunner::DataList outputs;
    MxnetRunner::DataList gradients;
  
    ASSERT_TRUE(Status::Ok() ==
                runner.Run(inputs, &outputs, &gradients));

    ASSERT_EQ(2, outputs.size());
    ASSERT_EQ(outputs[0].Size(), bs);
    ASSERT_EQ(outputs[1].Size(), bs);
    for (size_t i = 0; i < bs; ++i) {
      ASSERT_FLOAT_EQ(outputs[0].At(0, i), -0.5 + 0.5 * i);
      ASSERT_FLOAT_EQ(outputs[1].At(0, i), 3);
    }
    //ArrayEqual(outputs[0], {-0.5, 0});  // bs = 2
    //ArrayEqual(outputs[1], {3, 3});

    ASSERT_EQ(3, gradients.size());
    ASSERT_EQ(gradients[0].Size(), bs);
    ASSERT_EQ(gradients[1].Size(), bs);
    ASSERT_EQ(gradients[2].Size(), bs);
    for (size_t i = 0; i < bs; ++i) {
      ASSERT_FLOAT_EQ(gradients[0].At(0, i), -0.5);
      ASSERT_FLOAT_EQ(gradients[1].At(0, i), -0.5);
      ASSERT_FLOAT_EQ(gradients[2].At(0, i), 0.5);
    }
    //ArrayEqual(gradients[0], {-0.5, -0.5});  // bs = 2
    //ArrayEqual(gradients[1], {-0.5, -0.5});
    //ArrayEqual(gradients[2], {0.5, 0.5});

    delete[] a_ptr;
    delete[] b_ptr;
    delete[] y_ptr;
  }
}

