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
using std::vector;

TEST(RmspropUpdater, RmspropUpdater) {
  UdfRegistry* udf_registry = UdfRegistry::Get("RmspropUpdater");
  Udf* udf = udf_registry->Build(std::vector<size_t>({0, 1, 2, 3, 4, 5}), std::vector<size_t>({}));
  UdfContext* ctx = new UdfContext;
  Variable* var = new Variable(new Tensor(DataType::kFloat, TensorShape({4, 8}), new ConstantInitializer(0), true, 1), nullptr, "");
  ctx->SetVariable(var);
  vector<Slices> slices(1, Slices{.slice_size = 8, .slice_id = std::vector<size_t>({0, 2}), .dim_part = 1, .variable = var, .writable = true});
  vector<Tensor> grad(1, Tensor(DataType::kFloat, TensorShape({2, 8}), new ConstantInitializer(1)));
  ctx->SetData(0, new WrapperData<vector<Slices> >(slices), true); // slices
  ctx->SetData(1, new WrapperData<vector<Tensor> >(grad), true); // grad_tensor
  ctx->SetData(2, new WrapperData<vector<double> >(vector<double>{1}), true); // learning_rate
  ctx->SetData(3, new WrapperData<vector<double> >(vector<double>{0.99}), true); // decay
  ctx->SetData(4, new WrapperData<vector<double> >(vector<double>{0.0}), true); // alpha
  ctx->SetData(5, new WrapperData<vector<double> >(vector<double>{0.0}), true); // epsilon

  EXPECT_TRUE(udf->Run(ctx).IsOk());
  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(-10, *(var->GetData()->Raw<float>(0) + i));
    EXPECT_FLOAT_EQ(-10, *(var->GetData()->Raw<float>(2) + i));
    EXPECT_EQ(0, *(var->GetData()->Raw<float>(1) + i));
    EXPECT_EQ(0, *(var->GetData()->Raw<float>(3) + i));
  }
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(-17.088812, *(var->GetData()->Raw<float>(0) + i));
    EXPECT_FLOAT_EQ(-17.088812, *(var->GetData()->Raw<float>(2) + i));
    EXPECT_EQ(0, *(var->GetData()->Raw<float>(1) + i));
    EXPECT_EQ(0, *(var->GetData()->Raw<float>(3) + i));
  }

  vector<Slices> slices2(1, Slices{.slice_size = 8, .slice_id = std::vector<size_t>({1, (size_t)ps::HashMap::NOT_ADD_ID, 2}), .dim_part = -1, .variable = var, .writable = true});
  vector<Tensor> grad2(1, Tensor(DataType::kFloat, TensorShape({3, 8}), new ConstantInitializer(1)));
  ctx->SetData(1, new WrapperData<vector<Tensor> >(grad2), true);  
  ctx->SetData(0, new WrapperData<vector<Slices> >(slices2), true);
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(-17.088812, *(var->GetData()->Raw<float>(0) + i));
    EXPECT_FLOAT_EQ(-22.891302, *(var->GetData()->Raw<float>(2) + i));
    EXPECT_EQ(-10, *(var->GetData()->Raw<float>(1) + i));
    EXPECT_EQ(0, *(var->GetData()->Raw<float>(3) + i));
  }
  vector<Slices> slices3(1, Slices{.slice_size = 8, .slice_id = std::vector<size_t>({(size_t)ps::HashMap::NOT_ADD_ID, 1}), .dim_part = -1, .variable = var, .writable = false});
  ctx->SetData(0, new WrapperData<vector<Slices> >(slices3), true);
  ps::Status status = udf->Run(ctx);
  EXPECT_FALSE(status.IsOk());
  EXPECT_EQ(status.Msg(), "slice is not writable");

  vector<Tensor> grad3(1, Tensor(DataType::kDouble, TensorShape({2, 8}), new ConstantInitializer(2)));
  ctx->SetData(0, new WrapperData<vector<Slices> >(slices2), true);
  ctx->SetData(1, new WrapperData<vector<Tensor> >(grad3), true);
  status = udf->Run(ctx);
  EXPECT_FALSE(status.IsOk());
  EXPECT_EQ(status.Msg(), "grad should has same datatype with variable");    

  delete var;
  delete ctx;
  delete udf;
}

