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
  for (size_t s : shape)  size *= s;
  return size;
}

static mx_float *InitPtr(mxnet::cpp::Context &ctx, MxnetRunner::InputList &inputs,
                         const std::string &name, std::vector<size_t> &shape) {
  const size_t size = GetSize(shape);
  mx_float *ptr = new mx_float[size];
  for (size_t i = 0; i < size; ++i) {
    ptr[i] = 1.F + rand() / (RAND_MAX / 0.5);
  }
  Tensor tensor(DeviceSingleton::CpuInstance(), TensorShape(shape), DataType::kFloat);
  memcpy(tensor.Raw<float>(), ptr, size * sizeof(float));
  inputs.push_back({name, tensor});
  return ptr;
}

static uint32_t *InitIndicatorPtr(mxnet::cpp::Context &ctx, MxnetRunner::InputList &inputs,
                                  const std::string &name, std::vector<size_t> &shape,
                                  unsigned comm_bs, NDArrayHandle &handle) {
  const size_t size = GetSize(shape);
  uint32_t *ptr = new uint32_t[size];
  unsigned per = size / comm_bs;
  unsigned indicator = 0;
  for (size_t i = 0; i < size; ++i) {
    ptr[i] = indicator;
    if (--per == 0) {
      per = size / comm_bs;
      if (indicator < comm_bs - 1)  ++indicator;
    }
  }

  /*
  MXNDArrayCreateEx(shape.data(),
                    shape.size(),
                    ctx.GetDeviceType(),
                    ctx.GetDeviceId(),
                    false,
                    4,  // int32
                    &handle);
  mxnet::cpp::NDArray nd(handle);
  void* dst;
  MXNDArrayGetData(handle, &dst);
  memcpy(dst, ptr, size * sizeof(uint32_t));
  */
  Tensor tensor(DeviceSingleton::CpuInstance(), TensorShape(shape), DataType::kInt32);
  memcpy(tensor.Raw<uint32_t>(), ptr, size * sizeof(uint32_t));
  inputs.push_back({name, tensor});
  return ptr;
}

TEST(MxnetBackendDnnTest, RunnerTest) {
  MxnetRunner runner;
  std::string symbol_path = "../test_data/backend/mx_dnn.json";
  std::string graph_def = FileUtils::ReadLocalFile(symbol_path);
  ASSERT_TRUE(!graph_def.empty());
  ASSERT_TRUE(Status::Ok() == runner.Init(graph_def, "cpu"));

  mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu();

  srand((unsigned) time(nullptr));
  for (unsigned n = 0; n < 3; ++n) {
    unsigned comm_bs = 2 + rand() / (RAND_MAX / 4);
    unsigned bs = comm_bs + rand() / (RAND_MAX / 4);
    printf("==== comm_bs=%d, bs=%d ====\n", comm_bs, bs);

    MxnetRunner::InputList inputs;
    NDArrayHandle handle;
    std::vector<size_t> data_shape = {comm_bs, 16};
    std::vector<size_t> fc1_weight_shape = {12, 16};
    std::vector<size_t> fc1_bias_shape = {12};
    std::vector<size_t> fc2_weight_shape = {8, 12};
    std::vector<size_t> fc2_bias_shape = {8};
    std::vector<size_t> fc3_weight_shape = {2, 8};
    std::vector<size_t> fc3_bias_shape = {2};
    std::vector<size_t> softmax_label_shape = {comm_bs, 2};
    std::vector<size_t> fake_shape = {3, 3};
    std::vector<size_t> indicator_shape = {bs};
    mx_float *data_ptr = InitPtr(ctx, inputs, "indata", data_shape);
    mx_float *fc1_weight_ptr = InitPtr(ctx, inputs, "fc1_weight", fc1_weight_shape);
    mx_float *fc1_bias_ptr = InitPtr(ctx, inputs, "fc1_bias", fc1_bias_shape);
    mx_float *fc2_weight_ptr = InitPtr(ctx, inputs, "fc2_weight", fc2_weight_shape);
    mx_float *fc2_bias_ptr = InitPtr(ctx, inputs, "fc2_bias", fc2_bias_shape);
    mx_float *fc3_weight_ptr = InitPtr(ctx, inputs, "fc3_weight", fc3_weight_shape);
    mx_float *fc3_bias_ptr = InitPtr(ctx, inputs, "fc3_bias", fc3_bias_shape);
    mx_float *softmax_label_ptr = InitPtr(ctx, inputs, "clabel", softmax_label_shape);
    mx_float *fake_ptr = InitPtr(ctx, inputs, "fake", fake_shape);
    uint32_t *indicator_ptr = InitIndicatorPtr(ctx, inputs, "indicator", indicator_shape, comm_bs, handle);

    MxnetRunner::DataList outputs;
    MxnetRunner::DataList gradients;
  
    ASSERT_TRUE(Status::Ok() ==
                runner.Run(inputs, &outputs, &gradients));

    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(outputs[0].Size(), bs * 2);
    ASSERT_EQ(outputs[0].GetShape().size(), 2);
    ASSERT_EQ(outputs[0].GetShape()[0], bs);
    ASSERT_EQ(outputs[0].GetShape()[1], 2);

    ASSERT_EQ(9, gradients.size());
    ASSERT_EQ(gradients[0].Size(), GetSize(data_shape));
    ASSERT_EQ(gradients[1].Size(), GetSize(fc1_weight_shape));
    ASSERT_EQ(gradients[2].Size(), GetSize(fc1_bias_shape));
    ASSERT_EQ(gradients[3].Size(), GetSize(fc2_weight_shape));
    ASSERT_EQ(gradients[4].Size(), GetSize(fc2_bias_shape));
    ASSERT_EQ(gradients[5].Size(), GetSize(fc3_weight_shape));
    ASSERT_EQ(gradients[6].Size(), GetSize(fc3_bias_shape));
    ASSERT_EQ(gradients[7].Size(), GetSize(softmax_label_shape));
    ASSERT_EQ(gradients[8].Size(), GetSize(indicator_shape));

    delete[] data_ptr;
    delete[] fc1_weight_ptr;
    delete[] fc1_bias_ptr;
    delete[] fc2_weight_ptr;
    delete[] fc2_bias_ptr;
    delete[] fc3_weight_ptr;
    delete[] fc3_bias_ptr;
    delete[] softmax_label_ptr;
    delete[] fake_ptr;
    delete[] indicator_ptr;
  }
}

