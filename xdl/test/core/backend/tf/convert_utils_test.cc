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
#include "xdl/core/backend/tf/convert_utils.h"
#include "xdl/core/backend/device_singleton.h"

using namespace xdl;

static void TensorEqual(const tensorflow::Tensor& src, 
                        const std::vector<float>& dst) {
  ASSERT_EQ(src.NumElements(), dst.size());
  const float* src_ptr = src.flat<float>().data();
  for (size_t i = 0; i < dst.size(); ++i) {
    ASSERT_FLOAT_EQ(src_ptr[i], dst[i]);
  }
}

#define TEST_XDL2TF_TYPE(XDL_TYPE, TF_TYPE) \
  {                                         \
    DataType s = XDL_TYPE;                  \
    tensorflow::DataType d;                 \
    XDL2TF::ConvertType(s, &d);             \
    ASSERT_EQ(TF_TYPE, d);                  \
  }                                         \

#define TEST_TF2XDL_TYPE(TF_TYPE, XDL_TYPE) \
  {                                         \
    tensorflow::DataType s = TF_TYPE;       \
    DataType d;                             \
    TF2XDL::ConvertType(s, &d);             \
    ASSERT_EQ(XDL_TYPE, d);                 \
  }                                         \
  

TEST(XDL2TF, XDL2TFTest) {
  TEST_XDL2TF_TYPE(DataType::kInt8, tensorflow::DT_INT8);
  TEST_XDL2TF_TYPE(DataType::kInt16, tensorflow::DT_INT16);
  TEST_XDL2TF_TYPE(DataType::kInt32, tensorflow::DT_INT32);
  TEST_XDL2TF_TYPE(DataType::kInt64, tensorflow::DT_INT64);
  TEST_XDL2TF_TYPE(DataType::kFloat, tensorflow::DT_FLOAT);
  TEST_XDL2TF_TYPE(DataType::kDouble, tensorflow::DT_DOUBLE);

  {
    TensorShape s({1,2,3});
    tensorflow::TensorShape d;
    ASSERT_TRUE(XDL2TF::ConvertShape(s, &d) == Status::Ok());
    ASSERT_EQ(1, d.dim_size(0));
    ASSERT_EQ(2, d.dim_size(1));
    ASSERT_EQ(3, d.dim_size(2));
  }

  {  
    Tensor s(DeviceSingleton::CpuInstance(), TensorShape({2,2}), DataType::kFloat);
    float buf[4] = {1.0, 2.0, 3.0, 4.0};
    memcpy(s.Raw<float>(), buf, 4 * sizeof(float));
    tensorflow::Tensor d;
    ASSERT_TRUE(XDL2TF::ConvertTensor(s, &d) == Status::Ok());
    ASSERT_EQ(tensorflow::DT_FLOAT, d.dtype());
    ASSERT_EQ(2, d.shape().dim_size(0));
    ASSERT_EQ(2, d.shape().dim_size(1));
    TensorEqual(d, {1.0, 2.0, 3.0, 4.0});
  }
}

TEST(TF2XDL, TF2XDLTest) {
  TEST_TF2XDL_TYPE(tensorflow::DT_INT8, DataType::kInt8);  
  TEST_TF2XDL_TYPE(tensorflow::DT_INT16, DataType::kInt16);  
  TEST_TF2XDL_TYPE(tensorflow::DT_INT32, DataType::kInt32);  
  TEST_TF2XDL_TYPE(tensorflow::DT_INT64, DataType::kInt64);  
  TEST_TF2XDL_TYPE(tensorflow::DT_FLOAT, DataType::kFloat);  
  TEST_TF2XDL_TYPE(tensorflow::DT_DOUBLE, DataType::kDouble);  

  {
    tensorflow::TensorShape s;
    s.AddDim(1);
    s.AddDim(2);
    s.AddDim(3);
    TensorShape d;
    ASSERT_TRUE(TF2XDL::ConvertShape(s, &d) == Status::Ok());
    ASSERT_EQ(1, d[0]);
    ASSERT_EQ(2, d[1]);
    ASSERT_EQ(3, d[2]);
  }

  {
    tensorflow::Tensor s(tensorflow::DT_FLOAT, {4});
    auto ptr = s.flat<float>();
    for (int i = 0; i < 4; ++i) {
      ptr(i) = (float)i;
    }

    Tensor d;
    ASSERT_TRUE(TF2XDL::ConvertTensor(s, &d) == Status::Ok());
    ASSERT_EQ(DataType::kFloat, d.Type());
    ASSERT_EQ(1, d.Shape().Size());
    ASSERT_EQ(4, d.Shape()[0]);
    float* base = d.Raw<float>();
    for (int i = 0; i < 4; ++i) {
      ASSERT_FLOAT_EQ(float(i), base[i]);
    }
  }
}
