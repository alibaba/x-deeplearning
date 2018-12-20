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
#include "ps-plus/client/partitioner.h"

using ps::client::PartitionerContext;
using ps::VariableInfo;
using ps::Data;

TEST(PartitionerTest, PartitionerContext) {
  VariableInfo info;
  Data* data = new Data;
  PartitionerContext ctx(info);
  EXPECT_EQ(nullptr, ctx.GetData(0));
  ctx.SetData(1, data);
  EXPECT_EQ(nullptr, ctx.GetData(0));
  EXPECT_EQ(data, ctx.GetData(1));
  EXPECT_EQ(nullptr, ctx.GetData(2));
}
