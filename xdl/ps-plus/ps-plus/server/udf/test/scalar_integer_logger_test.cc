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
#include "ps-plus/common/data.h"
#include "ps-plus/server/udf.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/common/hashmap.h"

using ps::server::Udf;
using ps::server::UdfContext;
using ps::server::UdfRegistry;
using ps::server::Variable;
using ps::server::Slices;
using ps::initializer::ConstantInitializer;
using ps::Initializer;
using ps::DataType;
using ps::TensorShape;
using ps::Tensor;
using ps::Data;
using ps::WrapperData;
using ps::HashMap;
using std::vector;
using std::string;

TEST(ScalarIntegerLogger, ScalarIntegerLogger) {
  UdfRegistry* udf_registry = UdfRegistry::Get("ScalarIntegerLogger");
  Udf* udf = udf_registry->Build(std::vector<size_t>({0, 1, 2}), std::vector<size_t>({}));
  UdfContext* ctx = new UdfContext;
  Variable* var = new Variable(new Tensor(DataType::kFloat, TensorShape({4, 8}), new ConstantInitializer(5), true, 1), nullptr, "");
  ctx->SetVariable(var);
  vector<Slices> slices(1, Slices{.slice_size = 8, .slice_id = std::vector<size_t>({0, 2}), .dim_part = -1, .variable = var, .writable = true});
  vector<string> slot_name(1, "test");
  int64_t MAGIC_NUMBER = 8848;
  ctx->SetData(0, new WrapperData<vector<Slices> >(slices), true);
  ctx->SetData(1, new WrapperData<vector<string> >(slot_name), true);
  ctx->SetData(2, new WrapperData<vector<int64_t> >(vector<int64_t>{MAGIC_NUMBER}), true);


  EXPECT_TRUE(udf->Run(ctx).IsOk());

  Tensor* result;
  EXPECT_TRUE(var->GetExistSlot("test", &result).IsOk());
  EXPECT_EQ(MAGIC_NUMBER, *(result->Raw<int64_t>(0)));
  EXPECT_EQ(0, *(result->Raw<int64_t>(1)));
  EXPECT_EQ(MAGIC_NUMBER, *(result->Raw<int64_t>(2)));
  EXPECT_EQ(0, *(result->Raw<int64_t>(3)));

  delete var;
  delete ctx;
  delete udf;
}

