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
#include <time.h>

using namespace xdl;

static size_t GetSize(const std::vector<size_t> &shape) {
  size_t size = 1;
  for (unsigned s : shape)  size *= s;
  return size;
}

static mx_float *InitPtr(mxnet::cpp::Context &ctx, MxnetRunner::InputList &inputs,
                         const std::string &name, std::vector<size_t> &shape, mx_float value) {
  const size_t size = GetSize(shape);
  mx_float *ptr = new mx_float[size];
  for (size_t i = 0; i < size; ++i) {
    ptr[i] = value;
  }
  Tensor tensor(DeviceSingleton::CpuInstance(), TensorShape(shape), DataType::kFloat);
  memcpy(tensor.Raw<float>(), ptr, size * sizeof(float));
  inputs.push_back({name, tensor});
  return ptr;
}

TEST(MxnetBackendDummyTest, RunnerTest) {
  MxnetRunner runner;
  std::string symbol_path = "../test_data/backend/mx_dummy.json";
  std::string graph_def = FileUtils::ReadLocalFile(symbol_path);
  ASSERT_TRUE(!graph_def.empty());
  ASSERT_TRUE(Status::Ok() == runner.Init(graph_def, "cpu"));

  mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu();

  unsigned bs = 4;

  MxnetRunner::InputList inputs;
  std::vector<size_t> data_shape = {bs, 2};
  std::vector<size_t> fc1_weight_shape = {1, 2};
  std::vector<size_t> fc1_bias_shape = {1};
  std::vector<size_t> softmax_label_shape = {bs, 1};
  mx_float *data_ptr = InitPtr(ctx, inputs, "indata", data_shape, 1.F);
  mx_float *fc1_weight_ptr = InitPtr(ctx, inputs, "fc1_weight", fc1_weight_shape, 1.F);
  mx_float *fc1_bias_ptr = InitPtr(ctx, inputs, "fc1_bias", fc1_bias_shape, 0.F);
  mx_float *softmax_label_ptr = InitPtr(ctx, inputs, "clabel", softmax_label_shape, 0.F);

  for (unsigned n = 0; n < 2; ++n) {
    printf("==== n=%u ==== \n", n);
    MxnetRunner::DataList outputs;
    MxnetRunner::DataList gradients;
  
    ASSERT_TRUE(Status::Ok() ==
                runner.Run(inputs, &outputs, &gradients));

    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(outputs[0].Size(), bs * 1);
    ASSERT_EQ(outputs[0].GetShape().size(), 2);
    ASSERT_EQ(outputs[0].GetShape()[0], bs);
    ASSERT_EQ(outputs[0].GetShape()[1], 1);
    for (unsigned i = 0; i < bs; ++i) {
      printf("  prop[%u] = %f\n", i, outputs[0].At(i, 0));
    }

    ASSERT_EQ(4, gradients.size());
    ASSERT_EQ(gradients[0].Size(), GetSize(data_shape));
    ASSERT_EQ(gradients[1].Size(), GetSize(fc1_weight_shape));
    ASSERT_EQ(gradients[2].Size(), GetSize(fc1_bias_shape));
    ASSERT_EQ(gradients[3].Size(), GetSize(softmax_label_shape));
    for (unsigned i = 0; i < bs; ++i) {
      printf("  din_grad[%u] = %f %f\n", i, gradients[0].At(i, 0), gradients[0].At(i, 1));
    }
    printf("  fc1_weight_grad = %f %f\n", gradients[1].At(0, 0), gradients[1].At(0, 1));
    printf("  fc1_bias_grad = %f\n", gradients[2].At(0, 0));
    for (unsigned i = 0; i < bs; ++i) {
      printf("  label_grad[%u] = %f\n", i, gradients[3].At(i, 0));
    }
  }

  delete[] data_ptr;
  delete[] fc1_weight_ptr;
  delete[] fc1_bias_ptr;
  delete[] softmax_label_ptr;
}

