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
#include "ps-plus/common/initializer/uniform_unit_scaling_initializer.h"

using ps::DataType;
using ps::Initializer;
using ps::initializer::UniformUnitScalingInitializer;
using ps::TensorShape;

TEST(UniformUnitScalingInitializer, UniformUnitScalingInitializer) {
  TensorShape shape({2, 8});
  Initializer* init = new UniformUnitScalingInitializer(shape, -1, 0);
  Initializer* init_ = new UniformUnitScalingInitializer(std::move(shape), -1, 0);
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
}

