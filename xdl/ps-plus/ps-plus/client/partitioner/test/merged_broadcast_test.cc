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
#include "ps-plus/client/partitioner/merged_broadcast.h"

using ps::client::PartitionerContext;
using ps::client::MergedPartitionerContext;
using ps::client::partitioner::MergedBroadcast;
using ps::VariableInfo;
using ps::Data;
using ps::WrapperData;

TEST(MergedBroadcastPartitionerTest, MergedBroadcastPartitionerTest) {
  VariableInfo info1;
  info1.parts.push_back(VariableInfo::Part{.server = 0, .size = 1});
  info1.parts.push_back(VariableInfo::Part{.server = 1, .size = 1});
  info1.parts.push_back(VariableInfo::Part{.server = 2, .size = 1});
  info1.parts.push_back(VariableInfo::Part{.server = 3, .size = 1});
  PartitionerContext* ctx1 = new PartitionerContext(info1);
  VariableInfo info2;
  info2.parts.push_back(VariableInfo::Part{.server = 0, .size = 1});
  info2.parts.push_back(VariableInfo::Part{.server = 1, .size = 1});
  info2.parts.push_back(VariableInfo::Part{.server = 2, .size = 1});
  info2.parts.push_back(VariableInfo::Part{.server = 3, .size = 1});
  PartitionerContext* ctx2 = new PartitionerContext(info2);
  WrapperData<int>* data = new WrapperData<int>(10);
  std::vector<Data*> result;
  MergedBroadcast splitter; 
  MergedPartitionerContext ctx;
  ctx.AddContext(ctx1);
  ctx.AddContext(ctx2);
  EXPECT_TRUE(splitter.Init(&ctx, data).IsOk());
  EXPECT_TRUE(splitter.Split(&ctx, data, &result).IsOk());
  EXPECT_EQ(4u, result.size());
  EXPECT_EQ(10, dynamic_cast<WrapperData<int>*>(result[0])->Internal());
  EXPECT_EQ(10, dynamic_cast<WrapperData<int>*>(result[1])->Internal());
  EXPECT_EQ(10, dynamic_cast<WrapperData<int>*>(result[2])->Internal());
  EXPECT_EQ(10, dynamic_cast<WrapperData<int>*>(result[3])->Internal());
  delete data;
}

