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

using ps::server::UdfContext;
using ps::server::Udf;
using ps::server::UdfRegistry;
using ps::Status;
using ps::Data;
using ps::server::StorageManager;

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
    Data* input;
    if (GetInput(ctx, 1, &input).IsOk()) {
      return Status::IndexOverflow("Check Error");
    }
    std::vector<Data*> inputs;
    PS_CHECK_STATUS(GetInput(ctx, 0, &input));
    PS_CHECK_STATUS(GetInputs(ctx, &inputs));
    PS_CHECK_STATUS(SetOutput(ctx, 0, new MockUdfData(dynamic_cast<MockUdfData*>(input)->id + 1)));
    if (SetOutput(ctx, 1, new MockUdfData(dynamic_cast<MockUdfData*>(input)->id + 1)).IsOk()) {
      return Status::IndexOverflow("Check Error");
    }
    if (inputs.size() != 1 || inputs[0] != input) {
      return Status::IndexOverflow("Check Error");
    }
    return Status::Ok();
  }
};

}

UDF_REGISTER(MockUdf, UdfTest_MockUdf, 1, 1);

TEST(UdfTest, Register) {
  UdfRegistry* mock = UdfRegistry::Get("UdfTest_MockUdf");
  UdfRegistry* error = UdfRegistry::Get("mock_udf");
  EXPECT_NE(nullptr, mock);
  EXPECT_EQ(nullptr, error);
  EXPECT_EQ(1u, mock->InputSize());
  EXPECT_EQ(1u, mock->OutputSize());
}

TEST(UdfTest, InputOutput) {
  UdfRegistry* mock = UdfRegistry::Get("UdfTest_MockUdf");
  Udf* udf = mock->Build(std::vector<size_t>({0}), std::vector<size_t>({1}));
  UdfContext* ctx = new UdfContext;
  EXPECT_TRUE(ctx->SetData(0, new MockUdfData(42), true).IsOk());
  EXPECT_TRUE(udf->Run(ctx).IsOk());
  ctx->ProcessOutputs(std::vector<size_t>({1}));
  EXPECT_EQ(43, dynamic_cast<MockUdfData*>(ctx->Outputs()[0])->id);
  delete ctx;
  EXPECT_FALSE(MockUdfData::MemoryLeak());
}
