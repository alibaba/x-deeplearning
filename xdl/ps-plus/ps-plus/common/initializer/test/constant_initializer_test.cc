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
#include "ps-plus/common/initializer/constant_initializer.h"

using ps::DataType;
using ps::Initializer;
using ps::initializer::ConstantInitializer;

TEST(ConstantInitializer, ConstantInitializer) {
  Initializer* init = new ConstantInitializer(1);
  char x[8] = {0};
  init->Init(x, DataType::kInt8, 4);
  EXPECT_EQ(1, x[0]);
  EXPECT_EQ(1, x[1]);
  EXPECT_EQ(1, x[2]);
  EXPECT_EQ(1, x[3]);
  EXPECT_EQ(0, x[4]);
  EXPECT_EQ(0, x[5]);
  EXPECT_EQ(0, x[6]);
  EXPECT_EQ(0, x[7]);
  Initializer* init_clone = init->Clone();
  init_clone->Init(x, DataType::kInt8, 5);
  EXPECT_EQ(1, x[0]);
  EXPECT_EQ(1, x[1]);
  EXPECT_EQ(1, x[2]);
  EXPECT_EQ(1, x[3]);
  EXPECT_EQ(1, x[4]);
  EXPECT_EQ(0, x[5]);
  EXPECT_EQ(0, x[6]);
  EXPECT_EQ(0, x[7]);
  size_t large_data = 1 << 16;
  char y[large_data] = {0};
  init->MultiThreadInit(y, DataType::kInt8, large_data);
  EXPECT_EQ(1, y[0]);
  EXPECT_EQ(1, y[large_data - 1]);
}

