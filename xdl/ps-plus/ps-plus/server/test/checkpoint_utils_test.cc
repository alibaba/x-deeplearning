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

#define private public
#include "ps-plus/server/checkpoint_utils.h"
#include "ps-plus/scheduler/scheduler_impl.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include <ps-plus/common/logging.h>
#undef private

using ps::Tensor;
using ps::Initializer;
using ps::DataType;
using ps::TensorShape;
using ps::HashMap;
using ps::HashMapImpl;
using ps::Hash128Key;
using ps::server::Variable;
using ps::server::CheckpointUtils;
using ps::initializer::ConstantInitializer;
using ps::WrapperData;
using ps::VariableInfoCollection;
using ps::VariableInfo;

TEST(CheckpointUtilsTest, CheckpointUtilsTest) {
  std::unordered_map<std::string, std::unique_ptr<Variable>> a;
  std::unordered_map<std::string, std::unique_ptr<Variable>> b;
  WrapperData<std::unique_ptr<HashMap> >* y_slicer = new WrapperData<std::unique_ptr<HashMap> >(new HashMapImpl<Hash128Key>(10));
  int64_t y_keys[] = {1, 2, 3, 4};
  std::vector<size_t> y_ids;
  size_t filtered;
  EXPECT_EQ(2, y_slicer->Internal()->Get(y_keys, 2, false, 1.0, &y_ids, nullptr, &filtered));
  EXPECT_EQ(2u, y_ids.size());
  EXPECT_EQ(1, y_ids[1] + y_ids[0]);
  a["x"].reset(new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(0)), new WrapperData<size_t>(10), "x"));
  a["y"].reset(new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(1)), y_slicer, "y"));
  a["z"].reset(new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(2)), new WrapperData<size_t>(5), "z"));
  a["x"]->GetVariableLikeSlot("slot", DataType::kInt16, []{return new ConstantInitializer(42);});
  VariableInfoCollection from = {.infos = {
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
    .type = VariableInfo::kHash128,
    .name = "y",
    .parts = {VariableInfo::Part{.server = 0, .size = 1}, {.server = 1, .size = 65534}},
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
  CheckpointUtils ckpt(from);

  ps::Status status = ckpt.SaveVariables(1, "memory://save", a);
  std::cout << status.ToString() << std::endl;
  EXPECT_TRUE(status.IsOk());

  VariableInfoCollection to = from;
  for (auto& info : to.infos) {
    info.args[VariableInfo::ORIGIN_FILE_PATH] = "memory://save";
    info.args[VariableInfo::ORIGIN_NAME] = info.name;
    //z=>d, y=>c, etc
    info.name[0] -= 22;
  }
  EXPECT_FALSE(ckpt.LoadVariables(to, 1, &b).IsOk());
  to.infos.erase(to.infos.begin());

  status = ckpt.LoadVariables(to, 1, &b);
  std::cout << status.ToString() << std::endl;
  EXPECT_TRUE(status.IsOk());

  EXPECT_TRUE(b.find("b") != b.end());
  EXPECT_TRUE(b.find("c") != b.end());
  EXPECT_TRUE(b.find("d") == b.end());
  EXPECT_TRUE(b.find("a") == b.end());
  Tensor* slot = b["b"]->GetVariableLikeSlot("slot", DataType::kInt16, []{return new ConstantInitializer(43);});
  EXPECT_EQ(TensorShape({4, 8}), b["b"]->GetData()->Shape());
  TensorShape c_shape = b["c"]->GetData()->Shape();
  EXPECT_EQ(TensorShape({ps::Tensor::DEFAULT_SEGMENT_SIZE, 8}), c_shape);
  EXPECT_EQ(TensorShape({4, 8}), slot->Shape());
  EXPECT_EQ(DataType::kInt8, b["b"]->GetData()->Type());
  EXPECT_EQ(DataType::kInt8, b["c"]->GetData()->Type());
  EXPECT_EQ(DataType::kInt16, slot->Type());
  for (size_t i = 0; i < 32; i++) {
    EXPECT_EQ(0, b["b"]->GetData()->Raw<int8_t>()[i]);
    EXPECT_EQ(1, b["c"]->GetData()->Raw<int8_t>()[i]);
    EXPECT_EQ(42, slot->Raw<int16_t>()[i]);
  }

  WrapperData<size_t>* x_slicer = dynamic_cast<WrapperData<size_t>*>(b["b"]->GetSlicer());
  WrapperData<std::unique_ptr<HashMap> >* c_slicer = dynamic_cast<WrapperData<std::unique_ptr<HashMap> >*>(b["c"]->GetSlicer());
  EXPECT_TRUE(x_slicer != nullptr);
  EXPECT_EQ(10u, x_slicer->Internal());
  EXPECT_TRUE(c_slicer != nullptr);
  int64_t keys1[] = {1, 2, 3, 4, 13, 14};
  tbb::concurrent_vector<size_t> y_reused_ids;
  size_t filter_count;
  EXPECT_EQ(3, y_slicer->Internal()->Get(keys1, 3, false, 1.0, &y_ids, &y_reused_ids, &filter_count));
  EXPECT_EQ(3u, y_ids.size());
  EXPECT_EQ(3, y_ids[2] + y_ids[1] + y_ids[0]);
  EXPECT_EQ(0u, y_reused_ids.size());
}

TEST(CheckpointUtilsTest, CheckpointUtilsDebug) {
}
