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
#include "ps-plus/common/hashmap.h"
#include "ps-plus/common/qrw_lock.h"
#include "ps-plus/server/udf.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include <iostream>

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
using ps::QRWLocker;
using ps::Status;

TEST(BuildHashSlice, BuildHashSlice) {
  UdfRegistry* udf_registry = UdfRegistry::Get("BuildHashSlice");
  Udf* udf = udf_registry->Build(std::vector<size_t>({0, 1, 2}), std::vector<size_t>({3}));
  UdfContext* ctx = new UdfContext;
  Variable* var = new Variable(new Tensor(DataType::kInt8, TensorShape({2, 8}), new ConstantInitializer(1)), new WrapperData<HashMap>(2));
  QRWLocker* locker = new QRWLocker(var->VariableLock(), QRWLocker::kSimpleRead);
  ctx->SetLocker(locker);
  Tensor d(DataType::kInt64, TensorShape({3, 2}), new ConstantInitializer(0));
  d.Raw<int64_t>()[0] = d.Raw<int64_t>()[1] = 10;
  d.Raw<int64_t>()[2] = d.Raw<int64_t>()[3] = 10;
  d.Raw<int64_t>()[4] = d.Raw<int64_t>()[5] = 12;
  ctx->SetVariable(var);
  EXPECT_TRUE(ctx->SetData(0, new WrapperData<Tensor>(d), true).IsOk());
  EXPECT_TRUE(ctx->SetData(1, new WrapperData<bool>(false), true).IsOk());
  EXPECT_TRUE(ctx->SetData(2, new WrapperData<double>(1.0), true).IsOk());
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  Data* output;
  EXPECT_TRUE(ctx->GetData(3, &output).IsOk());
  Slices slices = dynamic_cast<WrapperData<Slices>*>(output)->Internal();
  EXPECT_EQ(8u, slices.slice_size);
  EXPECT_EQ(3u, slices.slice_id.size());
  EXPECT_EQ(1u, slices.slice_id[0]);
  EXPECT_EQ(1u, slices.slice_id[1]);
  EXPECT_EQ(0u, slices.slice_id[2]);

  d.Raw<int64_t>()[0] = d.Raw<int64_t>()[1] = 12;
  d.Raw<int64_t>()[2] = d.Raw<int64_t>()[3] = 13;
  d.Raw<int64_t>()[4] = d.Raw<int64_t>()[5] = 14;
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  EXPECT_TRUE(ctx->GetData(3, &output).IsOk());
  slices = dynamic_cast<WrapperData<Slices>*>(output)->Internal();
  EXPECT_EQ(8u, slices.slice_size);
  EXPECT_EQ(3u, slices.slice_id.size());
  EXPECT_EQ(0u, slices.slice_id[0]);
  EXPECT_EQ(3u, slices.slice_id[1]);
  EXPECT_EQ(2u, slices.slice_id[2]);
  EXPECT_EQ(1, slices.dim_part);
  EXPECT_EQ(TensorShape({4, 8}), var->GetData()->Shape());
  EXPECT_EQ(0x0101010101010101l, var->GetData()->Raw<int64_t>()[0]);
  EXPECT_EQ(0x0101010101010101l, var->GetData()->Raw<int64_t>()[1]);
  EXPECT_EQ(0x0101010101010101l, var->GetData()->Raw<int64_t>()[2]);
  EXPECT_EQ(0x0101010101010101l, var->GetData()->Raw<int64_t>()[3]);

  EXPECT_TRUE(ctx->SetData(2, new WrapperData<double>(0.0), true).IsOk());
  d.Raw<int64_t>()[0] = d.Raw<int64_t>()[1] = 10;
  d.Raw<int64_t>()[2] = d.Raw<int64_t>()[3] = 12;
  d.Raw<int64_t>()[4] = d.Raw<int64_t>()[5] = 16;
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  EXPECT_TRUE(ctx->GetData(3, &output).IsOk());
  slices = dynamic_cast<WrapperData<Slices>*>(output)->Internal();
  EXPECT_EQ(8u, slices.slice_size);
  EXPECT_EQ(3u, slices.slice_id.size());
  EXPECT_EQ(1u, slices.slice_id[0]);
  EXPECT_EQ(0u, slices.slice_id[1]);
  EXPECT_EQ(HashMap::NOT_ADD_ID, slices.slice_id[2]);
  EXPECT_EQ(1, slices.dim_part);
  EXPECT_EQ(TensorShape({4, 8}), var->GetData()->Shape());

  ctx->SetVariable(nullptr);
  EXPECT_FALSE(udf->Run(ctx).IsOk());
  Variable* var1 = new Variable(new Tensor(DataType::kInt8, TensorShape({}), new ConstantInitializer(1)), new WrapperData<HashMap>(2));
  ctx->SetVariable(var1);
  EXPECT_FALSE(udf->Run(ctx).IsOk());
  Tensor d1(DataType::kInt64, TensorShape({3}), new ConstantInitializer(0));
  ctx->SetData(0, new WrapperData<Tensor>(d1), true);
  ctx->SetVariable(var);
  EXPECT_FALSE(udf->Run(ctx).IsOk());
  Variable* var2 = new Variable(new Tensor(DataType::kInt8, TensorShape({2, 8}), new ConstantInitializer(1)), nullptr);
  ctx->SetVariable(var2);
  EXPECT_FALSE(udf->Run(ctx).IsOk());

  delete locker;
  delete var;
  delete ctx;
  delete udf;
}

