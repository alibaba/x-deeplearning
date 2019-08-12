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
using ps::server::TensorSlices;
using ps::initializer::ConstantInitializer;
using ps::Initializer;
using ps::DataType;
using ps::TensorShape;
using ps::Tensor;
using ps::Data;
using ps::WrapperData;
using std::vector;

TEST(TransSlice, TransSlice) {
  UdfRegistry* udf_registry = UdfRegistry::Get("TransSlice");
  Udf* udf = udf_registry->Build(std::vector<size_t>({0}), std::vector<size_t>({1}));
  UdfContext* ctx = new UdfContext;
  Variable* var = new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(1)), nullptr, "");
  ctx->SetVariable(var);
  vector<Slices> slices;
  slices.push_back(Slices{.slice_size = 8, .slice_id = std::vector<size_t>({0, 2}), .dim_part = -1, .variable = var, .writable = true});
  EXPECT_TRUE(ctx->SetData(0, new WrapperData<vector<Slices> >(slices), true).IsOk());
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  Data* output;
  EXPECT_TRUE(ctx->GetData(1, &output).IsOk());
  TensorSlices& tensor_slices = dynamic_cast<WrapperData<TensorSlices>*>(output)->Internal();
  EXPECT_EQ(8u, tensor_slices.slice_size);
  EXPECT_EQ(2u, tensor_slices.slice_id.size());
  EXPECT_EQ(0u, tensor_slices.slice_id[0]);
  EXPECT_EQ(-1, tensor_slices.dim_part);
  EXPECT_EQ(0x0101010101010101l, tensor_slices.tensor.Raw<int64_t>()[0]);
  EXPECT_EQ(0x0101010101010101l, tensor_slices.tensor.Raw<int64_t>()[1]);
  EXPECT_EQ(0x0101010101010101l, tensor_slices.tensor.Raw<int64_t>()[2]);
  EXPECT_EQ(0x0101010101010101l, tensor_slices.tensor.Raw<int64_t>()[3]);
  delete var;
  delete ctx;
  delete udf;
}

