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

using ps::server::Udf;
using ps::server::UdfContext;
using ps::server::UdfRegistry;
using ps::server::Variable;
using ps::server::StorageManager;
using ps::server::Slices;
using ps::initializer::ConstantInitializer;
using ps::Initializer;
using ps::DataType;
using ps::TensorShape;
using ps::Tensor;
using ps::Data;
using ps::WrapperData;

TEST(IsInitialized, IsInitialized) {
  UdfRegistry* udf_registry = UdfRegistry::Get("IsInitialized");
  Udf* udf = udf_registry->Build(std::vector<size_t>({}), std::vector<size_t>({0}));
  StorageManager* sm = new StorageManager;
  UdfContext* ctx = new UdfContext;
  ctx->SetStorageManager(sm);
  sm->Set("a", [&]{ return new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(0)), nullptr, ""); });
  ctx->SetVariableName("a");
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  Data* output;
  EXPECT_TRUE(ctx->GetData(0, &output).IsOk());
  bool inited = dynamic_cast<WrapperData<bool>*>(output)->Internal();
  EXPECT_FALSE(inited);
  Variable* var;
  EXPECT_TRUE(sm->Get("a", &var).IsOk());
  var->SetRealInited(true);
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  output;
  EXPECT_TRUE(ctx->GetData(0, &output).IsOk());
  inited = dynamic_cast<WrapperData<bool>*>(output)->Internal();
  EXPECT_TRUE(inited);
  ctx->SetVariableName("b");
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  EXPECT_TRUE(ctx->GetData(0, &output).IsOk());
  inited = dynamic_cast<WrapperData<bool>*>(output)->Internal();
  EXPECT_FALSE(inited);
  delete ctx;
  delete sm;
  delete udf;
}

