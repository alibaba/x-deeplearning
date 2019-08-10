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
using ps::HashMapImpl;
using ps::QRWLocker;
using ps::Status;
using ps::Hash128Key;
using ps::server::StorageManager;

TEST(BuildHashSlice, BuildHashSlice) {
  UdfRegistry* udf_registry = UdfRegistry::Get("BuildHashSlice");
  Udf* udf = udf_registry->Build(std::vector<size_t>({0, 1, 2, 3, 4}), std::vector<size_t>({5}));
  UdfContext* ctx = new UdfContext;
  HashMap* hashmap = new HashMapImpl<Hash128Key>(2);
  Variable* var = new Variable(new Tensor(DataType::kInt8, TensorShape({1, 8}), new ConstantInitializer(1), true, 1), new WrapperData<std::unique_ptr<HashMap> >(hashmap), "");
  StorageManager* manager = new StorageManager();
  EXPECT_TRUE(manager->Set("var", [var](){return var;}).IsOk());

  ctx->SetStorageManager(manager);
  Tensor d(DataType::kInt64, TensorShape({3, 2}), new ConstantInitializer(0));
  d.Raw<int64_t>()[0] = d.Raw<int64_t>()[1] = 10;
  d.Raw<int64_t>()[2] = d.Raw<int64_t>()[3] = 10;
  d.Raw<int64_t>()[4] = d.Raw<int64_t>()[5] = 12;

  ctx->SetVariable(var);
  EXPECT_TRUE(ctx->SetData(0, new WrapperData<std::vector<Tensor> >(std::vector<Tensor>{d}), true).IsOk());
  EXPECT_TRUE(ctx->SetData(1, new WrapperData<std::vector<std::string> >(std::vector<std::string>{"var"}), true).IsOk());
  EXPECT_TRUE(ctx->SetData(2, new WrapperData<std::vector<float> >(std::vector<float>{1.0}), true).IsOk());
  EXPECT_TRUE(ctx->SetData(3, new WrapperData<bool>(false), true).IsOk());
  EXPECT_TRUE(ctx->SetData(4, new WrapperData<bool>(true), true).IsOk());

  ps::Status status = udf->Run(ctx);
  EXPECT_TRUE(status.IsOk());
  Data* output;
  EXPECT_TRUE(ctx->GetData(5, &output).IsOk());
  std::vector<Slices>& slices0 = dynamic_cast<WrapperData<std::vector<Slices> >*>(output)->Internal();
  EXPECT_EQ(1, slices0.size());
  EXPECT_EQ(8u, slices0[0].slice_size);
  EXPECT_EQ(3u, slices0[0].slice_id.size());
  EXPECT_EQ(1u, slices0[0].slice_id[0] + slices0[0].slice_id[2]);
  EXPECT_EQ(slices0[0].slice_id[0], slices0[0].slice_id[1]);

  d.Raw<int64_t>()[0] = d.Raw<int64_t>()[1] = 12;
  d.Raw<int64_t>()[2] = d.Raw<int64_t>()[3] = 13;
  d.Raw<int64_t>()[4] = d.Raw<int64_t>()[5] = 14;
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  EXPECT_TRUE(ctx->GetData(5, &output).IsOk());
  std::vector<Slices> slices1 = dynamic_cast<WrapperData<std::vector<Slices> >*>(output)->Internal();
  EXPECT_EQ(8u, slices1[0].slice_size);
  EXPECT_EQ(3u, slices1[0].slice_id.size());
  EXPECT_EQ(slices0[0].slice_id[2], slices1[0].slice_id[0]);
  EXPECT_EQ(5, slices1[0].slice_id[1] + slices1[0].slice_id[2]);
  EXPECT_EQ(1, slices1[0].dim_part);
  EXPECT_EQ(0x0101010101010101l, *var->GetData()->Raw<int64_t>(0));
  EXPECT_EQ(0x0101010101010101l, *var->GetData()->Raw<int64_t>(1));
  EXPECT_EQ(0x0101010101010101l, *var->GetData()->Raw<int64_t>(2));
  EXPECT_EQ(0x0101010101010101l, *var->GetData()->Raw<int64_t>(3));

  EXPECT_TRUE(ctx->SetData(3, new WrapperData<bool>(true), true).IsOk());
  EXPECT_TRUE(ctx->SetData(4, new WrapperData<bool>(false), true).IsOk());  
  d.Raw<int64_t>()[0] = d.Raw<int64_t>()[1] = 10;
  d.Raw<int64_t>()[2] = d.Raw<int64_t>()[3] = 12;
  d.Raw<int64_t>()[4] = d.Raw<int64_t>()[5] = 16;
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  EXPECT_TRUE(ctx->GetData(5, &output).IsOk());
  std::vector<Slices>& slices2 = dynamic_cast<WrapperData<std::vector<Slices> >*>(output)->Internal();
  EXPECT_EQ(8u, slices2[0].slice_size);
  EXPECT_EQ(3u, slices2[0].slice_id.size());
  EXPECT_EQ(slices0[0].slice_id[0], slices2[0].slice_id[0]);
  EXPECT_EQ(slices0[0].slice_id[2], slices2[0].slice_id[1]);
  EXPECT_EQ(0, slices2[0].slice_id[2]);
  EXPECT_EQ(1, slices2[0].dim_part);

  ctx->SetVariable(nullptr);
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  HashMap* hashmap2 = new HashMapImpl<Hash128Key>(2);
  Variable* var2 = new Variable(new Tensor(DataType::kInt8, TensorShape({}), new ConstantInitializer(1)), new WrapperData<HashMap*>(hashmap2), "");
  ctx->SetVariable(var2);
  EXPECT_TRUE(udf->Run(ctx).IsOk());

  delete manager;
  delete var2;
  delete ctx;
  delete udf;
}
