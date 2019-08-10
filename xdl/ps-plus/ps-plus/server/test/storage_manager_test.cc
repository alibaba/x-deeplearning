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
#include "ps-plus/server/storage_manager.h"
#include "ps-plus/common/initializer/constant_initializer.h"

using ps::Tensor;
using ps::Initializer;
using ps::DataType;
using ps::TensorShape;
using ps::server::Variable;
using ps::server::StorageManager;
using ps::initializer::ConstantInitializer;

TEST(StorageManagerTest, SingleThread) {
  StorageManager* manager = new StorageManager;
  Variable* rst = nullptr;
  Variable* x = new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(0)), nullptr, "");
  Variable* y = new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(0)), nullptr, "");
  Variable* z = new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(0)), nullptr, "");
  EXPECT_FALSE(manager->Get("abc", &rst).IsOk());
  EXPECT_TRUE(manager->Set("abc", [x](){return x;}).IsOk());
  EXPECT_FALSE(manager->Set("abc", [y](){return y;}).IsOk());
  EXPECT_FALSE(manager->Set("ABC", [](){return nullptr;}).IsOk());
  EXPECT_TRUE(manager->Set("ABC", [z](){return z;}).IsOk());
  EXPECT_TRUE(manager->Get("abc", &rst).IsOk());
  EXPECT_EQ(x, rst);
  EXPECT_TRUE(manager->Get("ABC", &rst).IsOk());
  EXPECT_EQ(z, rst);
  delete y;
  delete manager;
}

