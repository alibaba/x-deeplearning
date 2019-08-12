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

#include <stdio.h>

#include "gtest/gtest.h"
#include "ps-plus/scheduler/placementer.h"

using ps::scheduler::Placementer;
using ps::VariableInfo;
using ps::Status;
using ps::DataType;

TEST(BalancePlacementerV2Test, BalancePlacementerV2) {
  auto sp = ps::GetPlugin<Placementer>("BalanceV2");

  size_t sn = 2;
  std::vector<VariableInfo> input;
  std::vector<int64_t> shape1;
  shape1.push_back(4);
  shape1.push_back(8);
  std::vector<int64_t> shape2;
  shape2.push_back(4);
  shape2.push_back(8);
  std::vector<VariableInfo::Part> parts1;
  std::vector<VariableInfo::Part> parts2;
  std::unordered_map<std::string, std::string> args1;
  std::unordered_map<std::string, std::string> args2;

  VariableInfo info1{.type = VariableInfo::Type::kIndex, .name = "var1", .parts = parts1, .shape = shape1, .datatype = DataType::kInt8, .args = args1};
  VariableInfo info2{.type = VariableInfo::Type::kIndex, .name = "var2", .parts = parts2, .shape = shape2, .datatype = DataType::kInt8, .args = args2};
  input.push_back(info1);
  input.push_back(info2);
  std::vector<VariableInfo> output;
  Placementer::Arg arg{.net = 0, .mem = 0, .query = 0};

  Status st = sp->Placement(input, &output, arg, sn);
  EXPECT_EQ(0, output.size());
  std::vector<VariableInfo> input1 = output;
  EXPECT_TRUE(sp->Placement(input1, &output, arg, sn).IsOk());
  EXPECT_EQ(0, output.size());
  Placementer::Arg arg1{.net = 100, .mem = 10, .query = 100};
  EXPECT_FALSE(sp->Placement(input, &output, arg1, sn).IsOk());
}

