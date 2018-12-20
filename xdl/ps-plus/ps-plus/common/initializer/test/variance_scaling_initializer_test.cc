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
#include "ps-plus/common/tensor_shape.h"
#include "ps-plus/common/initializer/variance_scaling_initializer.h"

using ps::DataType;
using ps::Initializer;
using ps::initializer::VarianceScalingInitializer;
using ps::TensorShape;

TEST(VarianceScalingInitializer, VarianceScalingInitializer) {
  TensorShape shape({2, 8});
  Initializer* init = new VarianceScalingInitializer(shape, -1, 0, "fan_in", "normal");
  Initializer* init_ = new VarianceScalingInitializer(std::move(shape), -1, 0, "fan_out", "else");
  float x[8] = {0};
  init->Init(x, DataType::kFloat, 8);
  EXPECT_TRUE(init->Accept(DataType::kFloat));
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(0, x[i]);
  }
  double y[8] = {0};
  Initializer* init_clone = init->Clone();
  init_clone->Init(y, DataType::kDouble, 8);
  EXPECT_TRUE(init_clone->Accept(DataType::kDouble));
  EXPECT_FALSE(init_clone->Accept(DataType::kInt8));
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(0, y[i]);
  }

  init_->Init(x, DataType::kFloat, 8);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(0, x[i]);
  }
  init_->Init(y, DataType::kDouble, 8);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(0, y[i]);
  }
  TensorShape shape1({});
  Initializer* init1 = new VarianceScalingInitializer(shape1, -1, 0, "else", "normal");
  double z = 0;
  init1->Init(&z, DataType::kDouble, 1);
  EXPECT_EQ(0, z);
  TensorShape shape2({2});
  Initializer* init2 = new VarianceScalingInitializer(shape2, -1, 0, "else", "normal");
  double z2[2] = {0};
  init2->Init(z2, DataType::kDouble, 2);
  EXPECT_EQ(0, z2[0]);
  EXPECT_EQ(0, z2[1]);
  TensorShape shape3({2, 3, 4});
  Initializer* init3 = new VarianceScalingInitializer(shape3, -1, 0, "else", "normal");
  double z3[24] = {0};
  init3->Init(z3, DataType::kDouble, 24);
  for (size_t i = 0; i< 24; ++i) {
    EXPECT_EQ(0, z3[i]);
  }
}

