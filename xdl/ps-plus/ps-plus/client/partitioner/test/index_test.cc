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
#include "ps-plus/client/partitioner/index.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/initializer/none_initializer.h"

using ps::client::PartitionerContext;
using ps::client::partitioner::IndexDataType;
using ps::client::partitioner::IndexShape;
using ps::client::partitioner::IndexOffset;
using ps::VariableInfo;
using ps::Data;
using ps::WrapperData;
using ps::Tensor;
using ps::TensorShape;
using ps::DataType;
using ps::initializer::NoneInitializer;

TEST(IndexPartitionerTest, IndexDataTypePartitionerTest) {
  VariableInfo info;
  info.parts.push_back(VariableInfo::Part{.server = 0, .size = 1});
  info.parts.push_back(VariableInfo::Part{.server = 1, .size = 2});
  info.parts.push_back(VariableInfo::Part{.server = 2, .size = 3});
  info.parts.push_back(VariableInfo::Part{.server = 3, .size = 4});
  info.datatype = DataType::kInt64;
  PartitionerContext ctx(info);
  std::unique_ptr<WrapperData<Tensor>> data(new WrapperData<Tensor>(DataType::kInt64, TensorShape({10, 2}), new NoneInitializer)); 
  for (int i = 0; i < 20; i++) {
    data->Internal().Raw<int64_t>()[i] = i;
  }
  std::vector<Data*> result;
  IndexDataType partitioner;
  EXPECT_TRUE(partitioner.Init(&ctx, data.get()).IsOk());
  EXPECT_TRUE(partitioner.Split(&ctx, data.get(), &result).IsOk());
  EXPECT_EQ(4u, result.size());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<DataType>*>(result[0])->Internal());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<DataType>*>(result[1])->Internal());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<DataType>*>(result[2])->Internal());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<DataType>*>(result[3])->Internal());
}

TEST(IndexPartitionerTest, IndexShapePartitionerTest) {
  VariableInfo info;
  info.parts.push_back(VariableInfo::Part{.server = 0, .size = 1});
  info.parts.push_back(VariableInfo::Part{.server = 1, .size = 2});
  info.parts.push_back(VariableInfo::Part{.server = 2, .size = 3});
  info.parts.push_back(VariableInfo::Part{.server = 3, .size = 4});
  info.datatype = DataType::kInt64;
  info.shape.push_back(10);
  info.shape.push_back(2);
  PartitionerContext ctx(info);
  std::unique_ptr<WrapperData<Tensor>> data(new WrapperData<Tensor>(DataType::kInt64, TensorShape({10, 2}), new NoneInitializer)); 
  for (int i = 0; i < 20; i++) {
    data->Internal().Raw<int64_t>()[i] = i;
  }
  std::vector<Data*> result;
  IndexShape partitioner;
  EXPECT_TRUE(partitioner.Init(&ctx, data.get()).IsOk());
  EXPECT_TRUE(partitioner.Split(&ctx, data.get(), &result).IsOk());
  EXPECT_EQ(4u, result.size());
  EXPECT_EQ(TensorShape({1, 2}), dynamic_cast<WrapperData<TensorShape>*>(result[0])->Internal());
  EXPECT_EQ(TensorShape({2, 2}), dynamic_cast<WrapperData<TensorShape>*>(result[1])->Internal());
  EXPECT_EQ(TensorShape({3, 2}), dynamic_cast<WrapperData<TensorShape>*>(result[2])->Internal());
  EXPECT_EQ(TensorShape({4, 2}), dynamic_cast<WrapperData<TensorShape>*>(result[3])->Internal());
}

TEST(IndexPartitionerTest, IndexOffsetPartitionerTest) {
  VariableInfo info;
  info.parts.push_back(VariableInfo::Part{.server = 0, .size = 1});
  info.parts.push_back(VariableInfo::Part{.server = 1, .size = 2});
  info.parts.push_back(VariableInfo::Part{.server = 2, .size = 3});
  info.parts.push_back(VariableInfo::Part{.server = 3, .size = 4});
  info.datatype = DataType::kInt64;
  info.shape.push_back(10);
  info.shape.push_back(2);
  PartitionerContext ctx(info);
  std::unique_ptr<WrapperData<Tensor>> data(new WrapperData<Tensor>(DataType::kInt64, TensorShape({10, 2}), new NoneInitializer)); 
  for (int i = 0; i < 20; i++) {
    data->Internal().Raw<int64_t>()[i] = i;
  }
  std::vector<Data*> result;
  IndexOffset partitioner;
  EXPECT_TRUE(partitioner.Init(&ctx, data.get()).IsOk());
  EXPECT_TRUE(partitioner.Split(&ctx, data.get(), &result).IsOk());
  EXPECT_EQ(4u, result.size());
  EXPECT_EQ(0, dynamic_cast<WrapperData<size_t>*>(result[0])->Internal());
  EXPECT_EQ(1, dynamic_cast<WrapperData<size_t>*>(result[1])->Internal());
  EXPECT_EQ(3, dynamic_cast<WrapperData<size_t>*>(result[2])->Internal());
  EXPECT_EQ(6, dynamic_cast<WrapperData<size_t>*>(result[3])->Internal());
}
