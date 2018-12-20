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
#include "ps-plus/server/udf_context.h"

using ps::server::UdfContext;
using ps::Data;
using ps::QRWLocker;
using ps::server::Variable;
using ps::server::StorageManager;

namespace {

class MockUdfData : public Data {
 public:
  MockUdfData() {
    ref++;
  }
  ~MockUdfData() {
    ref--;
  }
  static bool MemoryLeak() {
    return ref != 0;
  }
  static int ref;
};

int MockUdfData::ref;

}

TEST(UdfContextTest, AddDependency) {
  UdfContext* ctx = new UdfContext;
  EXPECT_TRUE(ctx->AddDependency(new MockUdfData).IsOk());
  EXPECT_TRUE(ctx->AddDependency(new MockUdfData).IsOk());
  EXPECT_TRUE(ctx->AddDependency(new MockUdfData).IsOk());
  EXPECT_TRUE(ctx->AddDependency(new MockUdfData).IsOk());
  delete ctx;
  EXPECT_FALSE(MockUdfData::MemoryLeak());
}

TEST(UdfContextTest, Data) {
  UdfContext* ctx = new UdfContext;
  Data* d2 = new MockUdfData;
  EXPECT_TRUE(ctx->SetData(2, d2, true).IsOk());
  Data* d4 = new MockUdfData;
  EXPECT_TRUE(ctx->SetData(4, d4, false).IsOk());
  Data* d0 = new MockUdfData;
  EXPECT_TRUE(ctx->SetData(0, d0, true).IsOk());
  EXPECT_EQ(5u, ctx->DataSize());
  Data* rst;
  EXPECT_TRUE(ctx->GetData(0, &rst).IsOk());
  EXPECT_EQ(d0, rst);
  EXPECT_TRUE(ctx->GetData(2, &rst).IsOk());
  EXPECT_EQ(d2, rst);
  EXPECT_TRUE(ctx->GetData(4, &rst).IsOk());
  EXPECT_EQ(d4, rst);
  EXPECT_TRUE(ctx->GetData(3, &rst).IsOk());
  EXPECT_EQ(nullptr, rst);
  EXPECT_TRUE((ctx->GetData(5, &rst)).IsOk());
  EXPECT_EQ(nullptr, rst);
  EXPECT_EQ(6u, ctx->DataSize());
  delete ctx;
  delete d4;
  EXPECT_FALSE(MockUdfData::MemoryLeak());
}

TEST(UdfContextTest, Storage) {
  UdfContext* ctx = new UdfContext;
  int x;
  StorageManager* fake_ptr = reinterpret_cast<StorageManager*>(&x);
  EXPECT_EQ(nullptr, ctx->GetStorageManager());
  EXPECT_TRUE(ctx->SetStorageManager(fake_ptr).IsOk());
  EXPECT_EQ(fake_ptr, ctx->GetStorageManager());
  delete ctx;
}

TEST(UdfContextTest, Variable) {
  UdfContext* ctx = new UdfContext;
  int x;
  Variable* fake_ptr = reinterpret_cast<Variable*>(&x);
  EXPECT_EQ(nullptr, ctx->GetVariable());
  EXPECT_TRUE(ctx->SetVariable(fake_ptr).IsOk());
  EXPECT_EQ(fake_ptr, ctx->GetVariable());
  delete ctx;
}

TEST(UdfContextTest, Locker) {
  UdfContext* ctx = new UdfContext;
  int x;
  QRWLocker* fake_ptr = reinterpret_cast<QRWLocker*>(&x);
  EXPECT_EQ(nullptr, ctx->GetLocker());
  EXPECT_TRUE(ctx->SetLocker(fake_ptr).IsOk());
  EXPECT_EQ(fake_ptr, ctx->GetLocker());
  delete ctx;
}
