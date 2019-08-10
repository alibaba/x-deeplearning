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

#include <iostream>
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
using ps::Status;

TEST(AdagradUpdater, AdagradUpdater) {
  UdfRegistry* udf_registry = UdfRegistry::Get("AdagradUpdater");
  Udf* udf = udf_registry->Build(std::vector<size_t>({0, 1, 2, 3}), std::vector<size_t>({}));
  UdfContext* ctx = new UdfContext;
  Variable* var = new Variable(new Tensor(DataType::kFloat, TensorShape({4, 8}), new ConstantInitializer(5)), nullptr, "");
  ctx->SetVariable(var);
  std::vector<Slices> sv;
  sv.push_back(Slices{.slice_size = 8, .slice_id = std::vector<size_t>({0, 2}), .dim_part = 1, .variable = var, .writable = true});
  ctx->SetData(0, new WrapperData<std::vector<Slices> >(sv), true);
  std::vector<Tensor> tv;
  tv.push_back(Tensor(DataType::kFloat, TensorShape({2, 8}), new ConstantInitializer(2)));
  ctx->SetData(1, new WrapperData<std::vector<Tensor> >(tv), true);
  ctx->SetData(2, new WrapperData<std::vector<double> >(std::vector<double>{3}), true);
  ctx->SetData(3, new WrapperData<std::vector<double> >(std::vector<double>{5}), true);

  Status status = udf->Run(ctx);
  EXPECT_TRUE(status.IsOk());
  for (size_t i = 0; i < 8; i++) {
    EXPECT_EQ(3, var->GetData()->Raw<float>()[i]);
    EXPECT_EQ(3, var->GetData()->Raw<float>()[i + 16]);
    EXPECT_EQ(5, var->GetData()->Raw<float>()[i + 8]);
    EXPECT_EQ(5, var->GetData()->Raw<float>()[i + 24]);
  }
  std::vector<Tensor> tv2;
  tv2.push_back(Tensor(DataType::kFloat, TensorShape({2, 8}), new ConstantInitializer(4)));
  ctx->SetData(1, new WrapperData<std::vector<Tensor> >(tv2), true);
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(0.6, var->GetData()->Raw<float>()[i]);
    EXPECT_FLOAT_EQ(0.6, var->GetData()->Raw<float>()[i + 16]);
    EXPECT_EQ(5, var->GetData()->Raw<float>()[i + 8]);
    EXPECT_EQ(5, var->GetData()->Raw<float>()[i + 24]);
  }
  std::vector<Slices> slices2;
  slices2.push_back(Slices{.slice_size = 8, .slice_id = std::vector<size_t>({(size_t)ps::HashMap::NOT_ADD_ID, 1}), .dim_part = -1, .variable = var, .writable = true});
  ctx->SetData(0, new WrapperData<std::vector<Slices> >(slices2), true);
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(0.6, var->GetData()->Raw<float>()[i]);
    EXPECT_FLOAT_EQ(0.6, var->GetData()->Raw<float>()[i + 16]);
    EXPECT_FLOAT_EQ(2.381385, var->GetData()->Raw<float>()[i + 8]);
    EXPECT_EQ(5, var->GetData()->Raw<float>()[i + 24]);
  }  
  std::vector<Slices> slices3;
  slices3.push_back(Slices{.slice_size = 8, .slice_id = std::vector<size_t>({(size_t)ps::HashMap::NOT_ADD_ID, 1}), .dim_part = -1, .variable = var, .writable = false});
  ctx->SetData(0, new WrapperData<std::vector<Slices> >(slices3), true);
  status = udf->Run(ctx);
  EXPECT_FALSE(status.IsOk());
  EXPECT_EQ(status.Msg(), "slice is not writable");
  
  ctx->SetData(0, new WrapperData<std::vector<Slices> >(slices2), true);
  std::vector<Tensor> vt3(1, Tensor(DataType::kDouble, TensorShape({2, 8}), new ConstantInitializer(2)));
  ctx->SetData(1, new WrapperData<std::vector<Tensor> >(vt3), true);
  status = udf->Run(ctx);
  EXPECT_FALSE(status.IsOk());
  EXPECT_EQ(status.Msg(), "grad should has same datatype with variable");
  delete var;
  delete ctx;
  delete udf;
}

TEST(AdagradUpdater, AdagradUpdaterForHash) {
  UdfRegistry* udf_registry = UdfRegistry::Get("AdagradUpdater");
  Udf* udf = udf_registry->Build(std::vector<size_t>({0, 1, 2, 3}), std::vector<size_t>({}));
  UdfContext* ctx = new UdfContext;
  ps::server::StorageManager* manager = new ps::server::StorageManager();
  Variable* var0 = new Variable(new Tensor(DataType::kFloat, TensorShape({4, 8}), new ConstantInitializer(5), Tensor::TType::kSegment, true), nullptr, "");
  Variable* var1 = new Variable(new Tensor(DataType::kFloat, TensorShape({4, 8}), new ConstantInitializer(1), Tensor::TType::kContinuous, true), nullptr, "");
  EXPECT_TRUE(manager->Set("var0", [var0](){return var0;}).IsOk());
  EXPECT_TRUE(manager->Set("var1", [var1](){return var1;}).IsOk());  
  std::vector<Slices> slices;
  slices.push_back(Slices{.slice_size = 8, .slice_id = std::vector<size_t>({0, 2}), .dim_part = -1, .variable = var0, .writable = true});
  slices.push_back(Slices{.slice_size = 8, .slice_id = std::vector<size_t>({1, 2}), .dim_part = -1, .variable = var1, .writable = true});  
  ctx->SetData(0, new WrapperData<std::vector<Slices> >(slices), true);
  std::vector<Tensor> grads;
  grads.push_back(Tensor(DataType::kFloat, TensorShape({2, 8}), new ConstantInitializer(2)));
  grads.push_back(Tensor(DataType::kFloat, TensorShape({2, 8}), new ConstantInitializer(1)));  
  ctx->SetData(1, new WrapperData<std::vector<Tensor> >(grads), true);
  ctx->SetData(2, new WrapperData<std::vector<double> >(std::vector<double>{3, 4}), true);
  ctx->SetData(3, new WrapperData<std::vector<double> >(std::vector<double>{5, 3}), true);
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  for (size_t i = 0; i < 8; i++) {
    EXPECT_EQ(3, *(var0->GetData()->Raw<float>(0) + i));
    EXPECT_EQ(3, *(var0->GetData()->Raw<float>(2) + i));
    EXPECT_EQ(5, *(var0->GetData()->Raw<float>(1) + i));
    EXPECT_EQ(5, *(var0->GetData()->Raw<float>(3) + i));

    EXPECT_EQ(-1, *(var1->GetData()->Raw<float>(1) + i));
    EXPECT_EQ(-1, *(var1->GetData()->Raw<float>(2) + i));    
    EXPECT_EQ(1, *(var1->GetData()->Raw<float>(0) + i));
    EXPECT_EQ(1, *(var1->GetData()->Raw<float>(3) + i));
  }

  std::vector<Tensor> grads2;
  grads2.push_back(Tensor(DataType::kFloat, TensorShape({2, 8}), new ConstantInitializer(4)));
  grads2.push_back(Tensor(DataType::kFloat, TensorShape({2, 8}), new ConstantInitializer(3)));  
  ctx->SetData(1, new WrapperData<std::vector<Tensor> >(grads2), true);

  EXPECT_TRUE(udf->Run(ctx).IsOk());
  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(0.6, *(var0->GetData()->Raw<float>(0) + i));
    EXPECT_FLOAT_EQ(0.6, *(var0->GetData()->Raw<float>(2) + i));
    EXPECT_FLOAT_EQ(5, *(var0->GetData()->Raw<float>(1) + i));
    EXPECT_FLOAT_EQ(5, *(var0->GetData()->Raw<float>(3) + i));

    EXPECT_FLOAT_EQ(-4.3282, *(var1->GetData()->Raw<float>(1) + i));
    EXPECT_FLOAT_EQ(-4.3282, *(var1->GetData()->Raw<float>(2) + i));
    EXPECT_FLOAT_EQ(1, *(var1->GetData()->Raw<float>(0) + i));
    EXPECT_FLOAT_EQ(1, *(var1->GetData()->Raw<float>(3) + i));
  }
  std::vector<Slices> slices2;
  slices2.push_back(Slices{.slice_size = 8, .slice_id = std::vector<size_t>({(size_t)ps::HashMap::NOT_ADD_ID, 1}), .dim_part = -1, .variable = var0, .writable = true});
  slices2.push_back(Slices{.slice_size = 8, .slice_id = std::vector<size_t>({(size_t)ps::HashMap::NOT_ADD_ID, 1}), .dim_part = -1, .variable = var1, .writable = true});  
  ctx->SetData(0, new WrapperData<std::vector<Slices> >(slices2), true);
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(0.6, *(var0->GetData()->Raw<float>(0) + i));
    EXPECT_FLOAT_EQ(0.6, *(var0->GetData()->Raw<float>(2) + i));
    EXPECT_FLOAT_EQ(2.381385, *(var0->GetData()->Raw<float>(1) + i));
    EXPECT_FLOAT_EQ(5, *(var0->GetData()->Raw<float>(3) + i));

    EXPECT_FLOAT_EQ(-6.88661, *(var1->GetData()->Raw<float>(1) + i));
    EXPECT_FLOAT_EQ(-4.3282, *(var1->GetData()->Raw<float>(2) + i));
    EXPECT_FLOAT_EQ(1, *(var1->GetData()->Raw<float>(0) + i));
    EXPECT_FLOAT_EQ(1, *(var1->GetData()->Raw<float>(3) + i));
  }
  std::vector<Slices> slices3;
  slices3.push_back(Slices{.slice_size = 8, .slice_id = std::vector<size_t>({(size_t)ps::HashMap::NOT_ADD_ID, 1}), .dim_part = -1, .variable = var0, .writable = false});
  slices3.push_back(Slices{.slice_size = 8, .slice_id = std::vector<size_t>({(size_t)ps::HashMap::NOT_ADD_ID, 1}), .dim_part = -1, .variable = var1, .writable = true});
  ctx->SetData(0, new WrapperData<std::vector<Slices> >(slices3), true);
  Status status = udf->Run(ctx);
  EXPECT_FALSE(status.IsOk());
  EXPECT_EQ(status.Msg(), "slice is not writable");
  
  ctx->SetData(0, new WrapperData<std::vector<Slices> >(slices2), true);
  std::vector<Tensor> grad3;
  grad3.push_back(Tensor(DataType::kDouble, TensorShape({2, 8}), new ConstantInitializer(2)));
  grad3.push_back(Tensor(DataType::kFloat, TensorShape({2, 8}), new ConstantInitializer(2)));  
  ctx->SetData(1, new WrapperData<std::vector<Tensor> >(grad3), true);
  status = udf->Run(ctx);
  EXPECT_FALSE(status.IsOk());
  EXPECT_EQ(status.Msg(), "grad should has same datatype with variable");

  std::vector<Tensor> grad4;
  grad4.push_back(Tensor(DataType::kFloat, TensorShape({2, 8}), new ConstantInitializer(2)));
  ctx->SetData(1, new WrapperData<std::vector<Tensor> >(grad4), true);
  status = udf->Run(ctx);
  EXPECT_FALSE(status.IsOk());
  EXPECT_EQ(status.Msg(), "AdagradUpdater: slices and other size not match");  
  delete manager;
  delete ctx;
  delete udf;
}
