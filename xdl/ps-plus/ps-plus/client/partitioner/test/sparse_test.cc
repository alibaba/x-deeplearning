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
#include "ps-plus/client/partitioner/sparse.h"
#include "ps-plus/common/initializer/none_initializer.h"

using ps::client::PartitionerContext;
using ps::client::partitioner::SparseId;
using ps::client::partitioner::SparseData;
using ps::VariableInfo;
using ps::Data;
using ps::WrapperData;
using ps::Tensor;
using ps::TensorShape;
using ps::DataType;
using ps::initializer::NoneInitializer;

TEST(SparsePartitionerTest, SplitAndCombine) {
  VariableInfo info;
  info.parts.push_back(VariableInfo::Part{.server = 0, .size = 1});
  info.parts.push_back(VariableInfo::Part{.server = 1, .size = 2});
  info.parts.push_back(VariableInfo::Part{.server = 2, .size = 4});
  info.parts.push_back(VariableInfo::Part{.server = 3, .size = 3});
  info.parts.push_back(VariableInfo::Part{.server = 4, .size = 5});
  info.shape.push_back(15);
  info.type = VariableInfo::kIndex;
  info.datatype = DataType::kInt64;
  PartitionerContext ctx(info);

  std::unique_ptr<WrapperData<Tensor>> data(new WrapperData<Tensor>(DataType::kInt64, TensorShape({5, 2}), new NoneInitializer));
  for (int i = 0; i < 10; i++) {
    data->Internal().Raw<int64_t>()[i] = i;
  }

  std::unique_ptr<WrapperData<Tensor>> ids(new WrapperData<Tensor>(DataType::kInt64, TensorShape({5}), new NoneInitializer));
  int64_t ids_value[] = {2, 3, 6, 10, 11};
  for (int i = 0; i < 5; i++) {
    ids->Internal().Raw<int64_t>()[i] = ids_value[i];
  }

  std::vector<Data*> result;
  std::vector<Data*> id_result;
  SparseId id_partitioner;
  SparseData data_partitioner;

  EXPECT_TRUE(id_partitioner.Init(&ctx, ids.get()).IsOk());
  EXPECT_TRUE(id_partitioner.Split(&ctx, ids.get(), &id_result).IsOk());
  EXPECT_EQ(5u, id_result.size());

  EXPECT_EQ(TensorShape({0}), dynamic_cast<WrapperData<Tensor>*>(id_result[0])->Internal().Shape());
  EXPECT_EQ(TensorShape({1}), dynamic_cast<WrapperData<Tensor>*>(id_result[1])->Internal().Shape());
  EXPECT_EQ(TensorShape({2}), dynamic_cast<WrapperData<Tensor>*>(id_result[2])->Internal().Shape());
  EXPECT_EQ(TensorShape({0}), dynamic_cast<WrapperData<Tensor>*>(id_result[3])->Internal().Shape());
  EXPECT_EQ(TensorShape({2}), dynamic_cast<WrapperData<Tensor>*>(id_result[4])->Internal().Shape());

  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(id_result[0])->Internal().Type());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(id_result[1])->Internal().Type());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(id_result[2])->Internal().Type());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(id_result[3])->Internal().Type());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(id_result[4])->Internal().Type());

  EXPECT_EQ(2, dynamic_cast<WrapperData<Tensor>*>(id_result[1])->Internal().Raw<int64_t>()[0]);
  EXPECT_EQ(3, dynamic_cast<WrapperData<Tensor>*>(id_result[2])->Internal().Raw<int64_t>()[0]);
  EXPECT_EQ(6, dynamic_cast<WrapperData<Tensor>*>(id_result[2])->Internal().Raw<int64_t>()[1]);
  EXPECT_EQ(10, dynamic_cast<WrapperData<Tensor>*>(id_result[4])->Internal().Raw<int64_t>()[0]);
  EXPECT_EQ(11, dynamic_cast<WrapperData<Tensor>*>(id_result[4])->Internal().Raw<int64_t>()[1]);

  std::unique_ptr<Data> ids2;
  EXPECT_TRUE(id_partitioner.CombineInit(&ctx, &ids2).IsOk());
  EXPECT_TRUE(id_partitioner.Combine(&ctx, id_result[1], 1, &ids2).IsOk());
  EXPECT_TRUE(id_partitioner.Combine(&ctx, id_result[4], 4, &ids2).IsOk());
  EXPECT_TRUE(id_partitioner.Combine(&ctx, id_result[2], 2, &ids2).IsOk());
  EXPECT_EQ(TensorShape({5}), dynamic_cast<WrapperData<Tensor>*>(ids2.get())->Internal().Shape());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(ids2.get())->Internal().Type());
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(ids_value[i], dynamic_cast<WrapperData<Tensor>*>(ids2.get())->Internal().Raw<int64_t>()[i]);
  }


  EXPECT_TRUE(data_partitioner.Init(&ctx, data.get()).IsOk());
  EXPECT_TRUE(data_partitioner.Split(&ctx, data.get(), &result).IsOk());
  EXPECT_EQ(5u, result.size());

  EXPECT_EQ(TensorShape({0, 2}), dynamic_cast<WrapperData<Tensor>*>(result[0])->Internal().Shape());
  EXPECT_EQ(TensorShape({1, 2}), dynamic_cast<WrapperData<Tensor>*>(result[1])->Internal().Shape());
  EXPECT_EQ(TensorShape({2, 2}), dynamic_cast<WrapperData<Tensor>*>(result[2])->Internal().Shape());
  EXPECT_EQ(TensorShape({0, 2}), dynamic_cast<WrapperData<Tensor>*>(result[3])->Internal().Shape());
  EXPECT_EQ(TensorShape({2, 2}), dynamic_cast<WrapperData<Tensor>*>(result[4])->Internal().Shape());

  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(result[0])->Internal().Type());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(result[1])->Internal().Type());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(result[2])->Internal().Type());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(result[3])->Internal().Type());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(result[4])->Internal().Type());
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(i + 0, dynamic_cast<WrapperData<Tensor>*>(result[1])->Internal().Raw<int64_t>()[i]);
  }
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(i + 2, dynamic_cast<WrapperData<Tensor>*>(result[2])->Internal().Raw<int64_t>()[i]);
  }
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(i + 6, dynamic_cast<WrapperData<Tensor>*>(result[4])->Internal().Raw<int64_t>()[i]);
  }

  std::unique_ptr<Data> data2;
  info.shape.push_back(2);
  ctx.SetVariableInfo(info);
  EXPECT_TRUE(data_partitioner.CombineInit(&ctx, &data2).IsOk());
  EXPECT_TRUE(data_partitioner.Combine(&ctx, result[1], 1, &data2).IsOk());
  EXPECT_TRUE(data_partitioner.Combine(&ctx, result[4], 4, &data2).IsOk());
  EXPECT_TRUE(data_partitioner.Combine(&ctx, result[2], 2, &data2).IsOk());
  EXPECT_EQ(TensorShape({5, 2}), dynamic_cast<WrapperData<Tensor>*>(data2.get())->Internal().Shape());
  EXPECT_EQ(DataType::kInt64, dynamic_cast<WrapperData<Tensor>*>(data2.get())->Internal().Type());
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(i, dynamic_cast<WrapperData<Tensor>*>(data2.get())->Internal().Raw<int64_t>()[i]);
  }
}
