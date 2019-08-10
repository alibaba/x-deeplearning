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
#include "ps-plus/common/initializer/orthogonal_initializer.h"

using ps::DataType;
using ps::Initializer;
using ps::initializer::OrthogonalInitializer;

TEST(OrthogonalInitializer, OrthogonalInitializer) {
  float data[6] = {0};
  Initializer* init = new OrthogonalInitializer(3);
  init->Init(data, DataType::kFloat, 6);
  float sum = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    sum += data[i] * data[i+3];
  }

  static const float eps = 1e-6;
  EXPECT_TRUE(abs(sum) < eps);
}

