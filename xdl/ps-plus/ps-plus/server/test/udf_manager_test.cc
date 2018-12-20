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
#include "ps-plus/server/udf_manager.h"

using ps::server::UdfContext;
using ps::server::Udf;
using ps::server::UdfChain;
using ps::server::UdfChainManager;
using ps::UdfChainRegister;
using ps::Status;
using ps::Data;

namespace {

class MockUdfData : public Data {
 public:
  MockUdfData(int id_) {
    id = id_;
    ref++;
  }
  ~MockUdfData() {
    ref--;
  }
  static bool MemoryLeak() {
    return ref != 0;
  }
  int id;
  static int ref;
};

int MockUdfData::ref;

class MockUdf : public Udf {
 public:
  virtual Status Run(UdfContext* ctx) const {
    Data *input1, *input2;
    PS_CHECK_STATUS(GetInput(ctx, 0, &input1));
    PS_CHECK_STATUS(GetInput(ctx, 1, &input2));
    PS_CHECK_STATUS(SetOutput(ctx, 0,
          new MockUdfData(dynamic_cast<MockUdfData*>(input1)->id +
            dynamic_cast<MockUdfData*>(input2)->id)));
    return Status::Ok();
  }
};

UdfChainRegister BuildUdfChainRegister(int64_t hash) {
  UdfChainRegister def;
  def.hash = hash;
  UdfChainRegister::UdfDef s0, s1, s2, s3;
  s0.udf_name = "";
  s1.udf_name = "UdfManagerTest_MockUdf";
  s1.inputs.emplace_back(0, 0);
  s1.inputs.emplace_back(0, 1);
  s2.udf_name = "UdfManagerTest_MockUdf";
  s2.inputs.emplace_back(0, 2);
  s2.inputs.emplace_back(0, 3);
  s3.udf_name = "UdfManagerTest_MockUdf";
  s3.inputs.emplace_back(1, 0);
  s3.inputs.emplace_back(2, 0);
  def.udfs.push_back(s0);
  def.udfs.push_back(s1);
  def.udfs.push_back(s2);
  def.udfs.push_back(s3);
  def.outputs.emplace_back(3, 0);
  return def;
}

void AddInputs(UdfContext* ctx) {
  ctx->SetData(0, new MockUdfData(1), true);
  ctx->SetData(1, new MockUdfData(2), true);
  ctx->SetData(2, new MockUdfData(3), true);
  ctx->SetData(3, new MockUdfData(4), true);
}

}

UDF_REGISTER(MockUdf, UdfManagerTest_MockUdf, 2, 1);

TEST(UdfManagerTest, Build) {
  UdfChain chain;
  UdfContext* ctx = new UdfContext;
  AddInputs(ctx);

  EXPECT_TRUE(chain.BuildFromDef(BuildUdfChainRegister(0)).IsOk());
  EXPECT_TRUE(chain.Process(ctx).IsOk());
  EXPECT_EQ(1u, ctx->Outputs().size());
  EXPECT_EQ(10, dynamic_cast<MockUdfData*>(ctx->Outputs()[0])->id);

  delete ctx;
}

TEST(UdfManagerTest, Register) {
  UdfChainManager chain_manager;
  UdfContext* ctx = new UdfContext;
  AddInputs(ctx);

  EXPECT_TRUE(chain_manager.RegisterUdfChain(BuildUdfChainRegister(0)).IsOk());
  EXPECT_TRUE(chain_manager.RegisterUdfChain(BuildUdfChainRegister(1)).IsOk());
  EXPECT_TRUE(chain_manager.RegisterUdfChain(BuildUdfChainRegister(2)).IsOk());
  EXPECT_TRUE(chain_manager.RegisterUdfChain(BuildUdfChainRegister(3)).IsOk());
  EXPECT_TRUE(chain_manager.RegisterUdfChain(BuildUdfChainRegister(0)).IsOk());

  EXPECT_TRUE(chain_manager.GetUdfChain(0)->Process(ctx).IsOk());

  EXPECT_EQ(1u, ctx->Outputs().size());
  EXPECT_EQ(10, dynamic_cast<MockUdfData*>(ctx->Outputs()[0])->id);
  delete ctx;
}
