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
#include "xdl/core/backend/mxnet/mxnet_runner.h"
#include "xdl/core/backend/mxnet/convert_utils.h"
#include "xdl/core/backend/device_singleton.h"

using namespace xdl;

static void ArrayEqual(const mxnet::cpp::NDArray& src,
                       const std::vector<float>& dst) {
  ASSERT_EQ(src.Size(), dst.size());
  const float* src_ptr = src.GetData();
  for (size_t i = 0; i < dst.size(); ++i) {
    ASSERT_FLOAT_EQ(src_ptr[i], dst[i]);
  }
}

#define TEST_XDL2MX_TYPE(XDL_TYPE, MX_TYPE) \
  {                                         \
    DataType s = XDL_TYPE;                  \
    int d;                                  \
    XDL2MX::ConvertType(s, &d);             \
    ASSERT_EQ(MX_TYPE, d);                  \
  }                                         \

#define TEST_MX2XDL_TYPE(MX_TYPE, XDL_TYPE) \
  {                                         \
    int s = MX_TYPE;                        \
    DataType d;                             \
    MX2XDL::ConvertType(s, &d);             \
    ASSERT_EQ(XDL_TYPE, d);                 \
  }                                         \


TEST(XDL2MX, XDL2MXTest) {
  TEST_XDL2MX_TYPE(DataType::kInt8, 5);
  TEST_XDL2MX_TYPE(DataType::kInt32, 4);
  TEST_XDL2MX_TYPE(DataType::kInt64, 6);
  TEST_XDL2MX_TYPE(DataType::kFloat, 0);
  TEST_XDL2MX_TYPE(DataType::kDouble, 1);

  {
    TensorShape s({1, 2, 3});
    std::vector<mx_uint> d;
    ASSERT_TRUE(XDL2MX::ConvertShape(s, &d) == Status::Ok());
    ASSERT_EQ(1, d[0]);
    ASSERT_EQ(2, d[1]);
    ASSERT_EQ(3, d[2]);
  }

  {
    mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu();
    Tensor s(DeviceSingleton::CpuInstance(), TensorShape({2,2}), DataType::kFloat);
    float buf[4] = {1.0, 2.0, 3.0, 4.0};
    memcpy(s.Raw<float>(), buf, 4 * sizeof(float));
    mxnet::cpp::NDArray d;
    ASSERT_TRUE(XDL2MX::ConvertTensor(&ctx, s, &d) == Status::Ok());    
    ASSERT_TRUE(XDL2MX::CopyTensor(s, &d) == Status::Ok());
    ArrayEqual(d, {1.0, 2.0, 3.0, 4.0});
  }
}

TEST(MX2XDL, MX2XDLTest) { 
  TEST_MX2XDL_TYPE(5, DataType::kInt8);
  TEST_MX2XDL_TYPE(4, DataType::kInt32);
  TEST_MX2XDL_TYPE(6, DataType::kInt64);
  TEST_MX2XDL_TYPE(0, DataType::kFloat);
  TEST_MX2XDL_TYPE(1, DataType::kDouble);

  {
    std::vector<mx_uint> s{1, 2, 3};
    TensorShape d;
    ASSERT_TRUE(MX2XDL::ConvertShape(s, &d) == Status::Ok());
    ASSERT_EQ(1, d[0]);
    ASSERT_EQ(2, d[1]);
    ASSERT_EQ(3, d[2]);
  }

  {
    std::vector<mx_uint> shape = {4};
    mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu();
    NDArrayHandle handle;
    ASSERT_EQ(0, MXNDArrayCreateEx(shape.data(), 
                                   shape.size(), 
                                   ctx.GetDeviceType(),
                                   ctx.GetDeviceId(),
                                   false,
                                   0,
                                   &handle));
    mxnet::cpp::NDArray s(handle);
    void* data;
    ASSERT_EQ(0, MXNDArrayGetData(handle, &data));
    float* base = (float*)data;
    for (int i = 0; i < 4; ++i) {
      base[i] = float(i);
    }

    Tensor d;
    ASSERT_TRUE(MX2XDL::ConvertTensor(s, &d) == Status::Ok());
    ASSERT_EQ(DataType::kFloat, d.Type());
    ASSERT_EQ(1, d.Shape().Size());
    ASSERT_EQ(4, d.Shape()[0]);
    base = d.Raw<float>();
    for (int i = 0; i < 4; ++i) {
      ASSERT_FLOAT_EQ(float(i), base[i]);
    }
  }
}
