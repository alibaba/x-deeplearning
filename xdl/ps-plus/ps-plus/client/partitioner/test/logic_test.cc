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
#include "ps-plus/client/partitioner/logic.h"

using ps::client::PartitionerContext;
using ps::client::partitioner::Logic;
using ps::VariableInfo;
using ps::Data;
using ps::WrapperData;

TEST(LogicPartitionerTest, LogicPartitionerTest) {
  VariableInfo info;
  info.parts.push_back(VariableInfo::Part{.server = 0, .size = 1});
  info.parts.push_back(VariableInfo::Part{.server = 1, .size = 2});
  info.parts.push_back(VariableInfo::Part{.server = 2, .size = 3});
  info.parts.push_back(VariableInfo::Part{.server = 3, .size = 4});
  PartitionerContext ctx(info);
  Logic partitioner;
  WrapperData<bool>* data1 = new WrapperData<bool>(true);
  WrapperData<bool>* data2 = new WrapperData<bool>(false);
  WrapperData<bool>* data3 = new WrapperData<bool>(true);
  WrapperData<bool>* data4 = new WrapperData<bool>(false);
  //std::unique_ptr<WrapperData<bool>> data1(new WrapperData<bool>(true));
  //std::unique_ptr<WrapperData<bool>> data2(new WrapperData<bool>(false));
  //std::unique_ptr<WrapperData<bool>> data3(new WrapperData<bool>(true));
  //std::unique_ptr<WrapperData<bool>> data4(new WrapperData<bool>(false));

  std::unique_ptr<Data> output;

  EXPECT_TRUE(partitioner.Combine(&ctx, data1, 0, &output).IsOk());
  EXPECT_TRUE(dynamic_cast<WrapperData<bool>*>(output.get())->Internal());
  EXPECT_TRUE(partitioner.Combine(&ctx, data2, 1, &output).IsOk());
  EXPECT_FALSE(dynamic_cast<WrapperData<bool>*>(output.get())->Internal());
  EXPECT_TRUE(partitioner.Combine(&ctx, data3, 2, &output).IsOk());
  EXPECT_FALSE(dynamic_cast<WrapperData<bool>*>(output.get())->Internal());
  EXPECT_TRUE(partitioner.Combine(&ctx, data4, 3, &output).IsOk());
  EXPECT_FALSE(dynamic_cast<WrapperData<bool>*>(output.get())->Internal());
  delete data1;
  delete data2;
  delete data3;
  delete data4;
}
