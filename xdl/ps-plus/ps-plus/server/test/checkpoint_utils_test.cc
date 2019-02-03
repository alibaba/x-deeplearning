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
#include "ps-plus/server/checkpoint_utils.h"
#include "ps-plus/common/initializer/constant_initializer.h"

using ps::Tensor;
using ps::Initializer;
using ps::DataType;
using ps::TensorShape;
using ps::HashMap;
using ps::server::Variable;
using ps::server::CheckpointUtils;
using ps::initializer::ConstantInitializer;
using ps::WrapperData;
using ps::VariableInfoCollection;
using ps::VariableInfo;

TEST(CheckpointUtilsTest, CheckpointUtilsTest) {
  std::unordered_map<std::string, std::unique_ptr<Variable>> a;
  std::unordered_map<std::string, std::unique_ptr<Variable>> b;
  WrapperData<HashMap>* y_slicer = new WrapperData<HashMap>(10);
  int64_t y_keys[] = {1, 2, 3, 4}; 
  std::vector<int64_t> y_ids;
  std::vector<int64_t> y_reused_ids;
  EXPECT_EQ(0, y_slicer->Internal().Get(y_keys, 2, 2, &y_ids, &y_reused_ids));
  EXPECT_EQ(2u, y_ids.size());
  EXPECT_EQ(0, y_ids[1]);
  EXPECT_EQ(1, y_ids[0]);
  a["x"].reset(new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(0)), new WrapperData<size_t>(10)));
  a["y"].reset(new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(1)), y_slicer));
  a["z"].reset(new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(2)), new WrapperData<size_t>(5)));
  a["x"]->GetVariableLikeSlot("slot", DataType::kInt16, []{return new ConstantInitializer(42);});
  VariableInfoCollection infos = {.infos = {
      VariableInfo {
        .type = VariableInfo::kIndex,
        .name = "w",
        .parts = {VariableInfo::Part{.server = 0, .size = 10}, {.server = 1, .size = 4}},
        .shape = {14, 8},
        .datatype = DataType::kInt8,
        .args = {}},
      VariableInfo {
        .type = VariableInfo::kIndex,
        .name = "x",
        .parts = {VariableInfo::Part{.server = 0, .size = 10}, {.server = 1, .size = 4}},
        .shape = {14, 8},
        .datatype = DataType::kInt8,
        .args = {}},
      VariableInfo {
        .type = VariableInfo::kHash,
        .name = "y",
        .parts = {VariableInfo::Part{.server = 0, .size = 32760}, {.server = 1, .size = 32776}},
        .shape = {2, 8},
        .datatype = DataType::kInt8,
        .args = {}},
      VariableInfo {
        .type = VariableInfo::kIndex,
        .name = "z",
        .parts = {VariableInfo::Part{.server = 0, .size = 4}, {.server = 1, .size = 4}},
        .shape = {9, 8},
        .datatype = DataType::kInt8,
        .args = {{"save", "false"}}}
    }};
  CheckpointUtils ckpt("memory://save", infos);
  EXPECT_TRUE(ckpt.SaveVariables(1, a).IsOk());
  EXPECT_TRUE(ckpt.LoadVariables(infos, 1, &b).IsOk());
  EXPECT_TRUE(b.find("x") != b.end());
  EXPECT_TRUE(b.find("y") != b.end());
  EXPECT_TRUE(b.find("z") == b.end());
  EXPECT_TRUE(b.find("w") == b.end());
  Tensor* slot = b["x"]->GetVariableLikeSlot("slot", DataType::kInt16, []{return new ConstantInitializer(43);});
  EXPECT_EQ(4, b["x"]->GetData()->Shape()[0]);
  EXPECT_EQ(8, b["x"]->GetData()->Shape()[1]);
  EXPECT_EQ(12, b["y"]->GetData()->Shape()[0]);
  EXPECT_EQ(8, b["y"]->GetData()->Shape()[1]);
  EXPECT_EQ(TensorShape({4, 8}), slot->Shape());
  EXPECT_EQ(DataType::kInt8, b["x"]->GetData()->Type());
  EXPECT_EQ(DataType::kInt8, b["y"]->GetData()->Type());
  EXPECT_EQ(DataType::kInt16, slot->Type());
  for (size_t i = 0; i < 32; i++) {
    EXPECT_EQ(0, b["x"]->GetData()->Raw<int8_t>()[i]);
    EXPECT_EQ(1, b["y"]->GetData()->Raw<int8_t>()[i]);
    EXPECT_EQ(42, slot->Raw<int16_t>()[i]);
  }
  WrapperData<size_t>* x_slicer = dynamic_cast<WrapperData<size_t>*>(b["x"]->GetSlicer());
  y_slicer = dynamic_cast<WrapperData<HashMap>*>(b["y"]->GetSlicer());
  EXPECT_TRUE(x_slicer != nullptr);
  EXPECT_EQ(10u, x_slicer->Internal());
  EXPECT_TRUE(y_slicer != nullptr);
  int64_t keys1[] = {1, 2, 3, 4, 13, 14};
  EXPECT_EQ(0, y_slicer->Internal().Get(keys1, 3, 2, &y_ids, &y_reused_ids));
  EXPECT_EQ(3u, y_ids.size());
  EXPECT_EQ(3, y_ids[2] + y_ids[1] + y_ids[0]);
  EXPECT_EQ(0u, y_reused_ids.size());
}

