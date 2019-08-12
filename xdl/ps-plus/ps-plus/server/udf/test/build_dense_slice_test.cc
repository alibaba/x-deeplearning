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
using ps::server::Slices;
using ps::initializer::ConstantInitializer;
using ps::Initializer;
using ps::DataType;
using ps::TensorShape;
using ps::Tensor;
using ps::Data;
using ps::WrapperData;

TEST(BuildDenseSlice, BuildDenseSlice) {
  UdfRegistry* udf_registry = UdfRegistry::Get("BuildDenseSlice");
  Udf* udf = udf_registry->Build(std::vector<size_t>({0}), std::vector<size_t>({1}));
  UdfContext* ctx = new UdfContext;
  Variable* var = new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(0)), nullptr, "");
  ctx->SetVariable(var);
  EXPECT_TRUE(ctx->SetData(0, new WrapperData<bool>(false), true).IsOk());
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  Data* output;
  EXPECT_TRUE(ctx->GetData(1, &output).IsOk());
  std::vector<Slices>& slices = dynamic_cast<WrapperData<std::vector<Slices> >*>(output)->Internal();
  EXPECT_EQ(32u, slices[0].slice_size);
  EXPECT_EQ(1u, slices[0].slice_id.size());
  EXPECT_EQ(0u, slices[0].slice_id[0]);
  EXPECT_EQ(-1, slices[0].dim_part);
  ctx->SetVariable(nullptr);
  EXPECT_FALSE(udf->Run(ctx).IsOk());
  delete var;
  delete ctx;
  delete udf;
}
