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
#include "ps-plus/server/variable.h"
#include "ps-plus/common/initializer/constant_initializer.h"

using ps::TensorShape;
using ps::DataType;
using ps::Tensor;
using ps::DataType;
using ps::Initializer;
using ps::server::Variable;
using ps::initializer::ConstantInitializer;
using ps::QRWLocker;

TEST(VariableTest, Constructor) {
  Variable var(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(1)), nullptr, "");
  EXPECT_EQ(TensorShape({4, 8}), var.GetData()->Shape());
  EXPECT_EQ(DataType::kInt8, var.GetData()->Type());
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[0]);
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[1]);
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[2]);
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[3]);
}

TEST(VariableTest, Slot) {
  Variable var(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(1)), nullptr, "");
  QRWLocker locker(var.VariableLock(), QRWLocker::kSimpleRead);
  EXPECT_EQ(TensorShape({4, 8}), var.GetData()->Shape());
  EXPECT_EQ(DataType::kInt8, var.GetData()->Type());
  Tensor* x = var.GetSlot("slot1", [&]{ return var.VariableLikeSlot(DataType::kInt32, TensorShape({4, 8}), new ConstantInitializer(1)); });
  Tensor* y = var.GetSlot("slot1", [&]{ return var.VariableLikeSlot(DataType::kInt32, TensorShape({4, 8}), new ConstantInitializer(1)); });
  Tensor* z = var.GetSlot("slot2", [&]{ return Variable::Slot{
      .tensor = std::unique_ptr<Tensor>(new Tensor(DataType::kInt64, TensorShape({2, 8}), new ConstantInitializer(1))),
      .joiner = Variable::kAnyOne}; });
  EXPECT_EQ(x, y);
  EXPECT_EQ(TensorShape({4, 8}), x->Shape());
  EXPECT_EQ(DataType::kInt32, x->Type());
  EXPECT_EQ(TensorShape({2, 8}), z->Shape());
  EXPECT_EQ(DataType::kInt64, z->Type());
  var.ReShapeId(1);
  EXPECT_EQ(TensorShape({1, 8}), var.GetData()->Shape());
  EXPECT_EQ(DataType::kInt8, var.GetData()->Type());
  EXPECT_EQ(TensorShape({1, 8}), x->Shape());
  EXPECT_EQ(DataType::kInt32, x->Type());
  EXPECT_EQ(TensorShape({2, 8}), z->Shape());
  EXPECT_EQ(DataType::kInt64, z->Type());
}

TEST(VariableTest, ClearId) {
  Variable var(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(1), Tensor::TType::kSegment), nullptr, "");
  EXPECT_EQ(TensorShape({Tensor::DEFAULT_SEGMENT_SIZE, 8}), var.GetData()->Shape());
  EXPECT_EQ(DataType::kInt8, var.GetData()->Type());
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[0]);
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[1]);
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[2]);
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[3]);
  var.GetData()->Raw<int64_t>()[0] = 100;
  var.GetData()->Raw<int64_t>()[2] = 200;
  var.GetData()->Raw<int64_t>()[3] = 300;    
  EXPECT_EQ(100, var.GetData()->Raw<int64_t>()[0]);
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[1]);
  EXPECT_EQ(200, var.GetData()->Raw<int64_t>()[2]);
  EXPECT_EQ(300, var.GetData()->Raw<int64_t>()[3]);
  var.ClearIds(std::vector<size_t>{0, 2});
  /*
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[0]);
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[1]);
  EXPECT_EQ(0x0101010101010101, var.GetData()->Raw<int64_t>()[2]);
  EXPECT_EQ(300, var.GetData()->Raw<int64_t>()[3]);
  */
}
