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

TEST(SliceToTensor, SliceToTensor) {
  UdfRegistry* udf_registry = UdfRegistry::Get("SliceToTensor");
  Udf* udf = udf_registry->Build(std::vector<size_t>({0}), std::vector<size_t>({1}));
  UdfContext* ctx = new UdfContext;
  Variable* var = new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(1)), nullptr, "");
  ctx->SetVariable(var);
  std::vector<Slices> slices;
  slices.push_back(Slices{.slice_size = 32, .slice_id = std::vector<size_t>({0}), .dim_part = -1, .variable = var});
  EXPECT_TRUE(ctx->SetData(0, new WrapperData<std::vector<Slices> >(slices), true).IsOk());
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  Data* output;
  EXPECT_TRUE(ctx->GetData(1, &output).IsOk());
  std::vector<Tensor>& tensors = dynamic_cast<WrapperData<std::vector<Tensor> >*>(output)->Internal();
  EXPECT_EQ(1, tensors.size());
  TensorShape shape = tensors[0].Shape();
  EXPECT_EQ(2, shape.Size());
  EXPECT_EQ(4, shape.Dims()[0]);
  EXPECT_EQ(8, shape.Dims()[1]);
  EXPECT_EQ(0x0101010101010101l, *(tensors[0].Raw<int64_t>(0)));
  EXPECT_EQ(0x0101010101010101l, *(tensors[0].Raw<int64_t>(1)));
  EXPECT_EQ(0x0101010101010101l, *(tensors[0].Raw<int64_t>(2)));
  EXPECT_EQ(0x0101010101010101l, *(tensors[0].Raw<int64_t>(3)));

  Tensor* tensor_ = new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(1));
  Data* slicer_ = new WrapperData<size_t>(10);
  Variable* var1 = new Variable(tensor_, slicer_, "");
  Tensor* t = var1->GetData();
  ctx->SetVariable(var1);
  std::vector<Slices> slices1;
  slices1.push_back(Slices{.slice_size = 8, .slice_id = std::vector<size_t>({0, 2}), .dim_part = 1, .variable = var1});

  EXPECT_TRUE(ctx->SetData(0, new WrapperData<std::vector<Slices> >(slices1), true).IsOk());
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  EXPECT_TRUE(ctx->GetData(1, &output).IsOk());
  std::vector<Tensor>& tensors1 = dynamic_cast<WrapperData<std::vector<Tensor> >*>(output)->Internal();

  EXPECT_EQ(1u, tensors1.size());
  EXPECT_EQ(2u, tensors1[0].Shape().Size());
  EXPECT_EQ(2u, tensors1[0].Shape()[0]);
  EXPECT_EQ(8u, tensors1[0].Shape()[1]);
  EXPECT_EQ(0x0101010101010101l, *(tensors1[0].Raw<int64_t>(0)));
  EXPECT_EQ(0x0101010101010101l, *(tensors1[0].Raw<int64_t>(1)));
  TensorSlices slices2{.slice_size = 8, .slice_id = std::vector<size_t>({0, 2}), .dim_part = 3, .tensor = *(var1->GetData())};
  EXPECT_TRUE(ctx->SetData(0, new WrapperData<TensorSlices>(slices2), true).IsOk());
  EXPECT_FALSE(udf->Run(ctx).IsOk());

  delete var1;
  delete var;
  delete ctx;
  delete udf;
}
