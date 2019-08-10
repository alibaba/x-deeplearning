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
#include "ps-plus/client/partitioner/dense.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/initializer/none_initializer.h"

using ps::client::PartitionerContext;
using ps::client::partitioner::Dense;
using ps::VariableInfo;
using ps::Data;
using ps::WrapperData;
using ps::Tensor;
using ps::TensorShape;
using ps::DataType;
using ps::initializer::NoneInitializer;

TEST(DensePartitionerTest, SimpleSplitAndCombine) {
  VariableInfo info;
  info.parts.push_back(VariableInfo::Part{.server = 0, .size = 1});
  info.parts.push_back(VariableInfo::Part{.server = 1, .size = 2});
  info.parts.push_back(VariableInfo::Part{.server = 2, .size = 4});
  info.parts.push_back(VariableInfo::Part{.server = 3, .size = 3});
  info.shape.push_back(10);
  info.shape.push_back(2);
  info.type = VariableInfo::kIndex;
  info.datatype = DataType::kInt64;
  PartitionerContext ctx(info);
  std::unique_ptr<WrapperData<Tensor>> data(new WrapperData<Tensor>(DataType::kInt64, TensorShape({10, 2}), new NoneInitializer)); for (int i = 0; i < 20; i++) {
    data->Internal().Raw<int64_t>()[i] = i;
  }
  std::vector<Data*> result;
  Dense partitioner;
  EXPECT_TRUE(partitioner.Init(&ctx, data.get()).IsOk());
  EXPECT_TRUE(partitioner.Split(&ctx, data.get(), &result).IsOk());
  EXPECT_EQ(4u, result.size());
  EXPECT_EQ(TensorShape({1, 2}), dynamic_cast<WrapperData<Tensor>*>(result[0])->Internal().Shape());
  EXPECT_EQ(TensorShape({2, 2}), dynamic_cast<WrapperData<Tensor>*>(result[1])->Internal().Shape());
  EXPECT_EQ(TensorShape({4, 2}), dynamic_cast<WrapperData<Tensor>*>(result[2])->Internal().Shape());
  EXPECT_EQ(TensorShape({3, 2}), dynamic_cast<WrapperData<Tensor>*>(result[3])->Internal().Shape());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(result[0])->Internal().Type());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(result[1])->Internal().Type());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(result[2])->Internal().Type());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(result[3])->Internal().Type());
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(i + 0, dynamic_cast<WrapperData<Tensor>*>(result[0])->Internal().Raw<int64_t>()[i]);
  }
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(i + 2, dynamic_cast<WrapperData<Tensor>*>(result[1])->Internal().Raw<int64_t>()[i]);
  }
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(i + 6, dynamic_cast<WrapperData<Tensor>*>(result[2])->Internal().Raw<int64_t>()[i]);
  }
  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(i + 14, dynamic_cast<WrapperData<Tensor>*>(result[3])->Internal().Raw<int64_t>()[i]);
  }

  std::unique_ptr<Data> data2;
  EXPECT_TRUE(partitioner.CombineInit(&ctx, &data2).IsOk());
  EXPECT_TRUE(partitioner.Combine(&ctx, result[1], 1, &data2).IsOk());
  EXPECT_TRUE(partitioner.Combine(&ctx, result[3], 3, &data2).IsOk());
  EXPECT_TRUE(partitioner.Combine(&ctx, result[0], 0, &data2).IsOk());
  EXPECT_TRUE(partitioner.Combine(&ctx, result[2], 2, &data2).IsOk());
  EXPECT_EQ(TensorShape({10, 2}), dynamic_cast<WrapperData<Tensor>*>(data2.get())->Internal().Shape());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(data2.get())->Internal().Type());
  for (int i = 0; i < 20; i++) {
    EXPECT_EQ(i, dynamic_cast<WrapperData<Tensor>*>(data2.get())->Internal().Raw<int64_t>()[i]);
  }
}

TEST(DensePartitionerTest, ScalarSplitAndCombine) {
  VariableInfo info;
  info.parts.push_back(VariableInfo::Part{.server = 0, .size = 10});
  info.type = VariableInfo::kIndex;
  info.shape.push_back(10);
  info.shape.push_back(2);
  info.datatype = DataType::kInt64;
  PartitionerContext ctx(info);
  std::unique_ptr<WrapperData<Tensor>> data(new WrapperData<Tensor>(DataType::kInt64, TensorShape({10, 2}), new NoneInitializer));
  for (int i = 0; i < 20; i++) {
    data->Internal().Raw<int64_t>()[i] = i;
  }
  std::vector<Data*> result;
  Dense partitioner;
  EXPECT_TRUE(partitioner.Init(&ctx, data.get()).IsOk());
  EXPECT_TRUE(partitioner.Split(&ctx, data.get(), &result).IsOk());
  EXPECT_EQ(1u, result.size());
  EXPECT_EQ(TensorShape({10, 2}), dynamic_cast<WrapperData<Tensor>*>(result[0])->Internal().Shape());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(result[0])->Internal().Type());
  for (int i = 0; i < 20; i++) {
    EXPECT_EQ(i, dynamic_cast<WrapperData<Tensor>*>(result[0])->Internal().Raw<int64_t>()[i]);
  }
  std::unique_ptr<Data> data2;
  EXPECT_TRUE(partitioner.CombineInit(&ctx, &data2).IsOk());
  EXPECT_TRUE(partitioner.Combine(&ctx, result[0], 0, &data2).IsOk());
  EXPECT_EQ(TensorShape({10, 2}), dynamic_cast<WrapperData<Tensor>*>(data2.get())->Internal().Shape());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(data2.get())->Internal().Type());
  for (int i = 0; i < 20; i++) {
    EXPECT_EQ(i, dynamic_cast<WrapperData<Tensor>*>(data2.get())->Internal().Raw<int64_t>()[i]);
  }
}

