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
#include "ps-plus/server/server.h"
#include "ps-plus/server/local_server.h"
#include "ps-plus/message/variable_info.h"
#include "ps-plus/common/initializer/constant_initializer.h"

using ps::server::UdfContext;
using ps::server::Udf;
using ps::server::UdfChain;
using ps::server::UdfChainManager;
using ps::server::Server;
using ps::server::LocalServer;
using ps::server::Variable;
using ps::server::StreamingModelArgs;
using ps::initializer::ConstantInitializer;
using ps::UdfChainRegister;
using ps::Status;
using ps::Data;
using ps::Tensor;
using ps::WrapperData;
using ps::DataType;
using ps::TensorShape;
using ps::kUnusedVersion;
using ps::VariableInfoCollection;
using ps::DenseVarNames;
using ps::DenseVarValues;

namespace {

class MockUdf : public Udf {
 public:
  virtual Status Run(UdfContext* ctx) const {
    Data *input1, *input2;
    int x = 0;
    if (ctx->GetVariable() == nullptr) {
      x = 1;
    } else {
      x = 2;
    }
    PS_CHECK_STATUS(GetInput(ctx, 0, &input1));
    PS_CHECK_STATUS(GetInput(ctx, 1, &input2));
    PS_CHECK_STATUS(SetOutput(ctx, 0,
          new WrapperData<int>((dynamic_cast<WrapperData<int>*>(input1)->Internal() +
            dynamic_cast<WrapperData<int>*>(input2)->Internal()) * x)));
    return Status::Ok();
  }
};

UdfChainRegister BuildUdfChainRegister() {
  UdfChainRegister def;
  def.hash = 100;
  UdfChainRegister::UdfDef s0, s1, s2, s3;
  s0.udf_name = "";
  s1.udf_name = "ServerTest_MockUdf";
  s1.inputs.emplace_back(0, 0);
  s1.inputs.emplace_back(0, 1);
  s2.udf_name = "ServerTest_MockUdf";
  s2.inputs.emplace_back(0, 2);
  s2.inputs.emplace_back(0, 3);
  s3.udf_name = "ServerTest_MockUdf";
  s3.inputs.emplace_back(1, 0);
  s3.inputs.emplace_back(2, 0);
  def.udfs.push_back(s0);
  def.udfs.push_back(s1);
  def.udfs.push_back(s2);
  def.udfs.push_back(s3);
  def.outputs.emplace_back(3, 0);
  return def;
}

std::vector<Data*> Inputs() {
  return std::vector<Data*>({
      new WrapperData<int>(1),
      new WrapperData<int>(2),
      new WrapperData<int>(3),
      new WrapperData<int>(4)});
}

}

UDF_REGISTER(MockUdf, ServerTest_MockUdf, 2, 1);

TEST(ServerTest, ServerTest) {
  StreamingModelArgs args;
  args.streaming_dense_model_addr = "where?";
  Server server(0, args);
  Status st = server.Init();
  EXPECT_EQ(st, Status::Ok());

  VariableInfoCollection vic;
  st = server.Save(0, "hello", vic);
  EXPECT_EQ(st, Status::Ok());
  VariableInfoCollection from, to;
  st = server.Restore(0, from, to);
  EXPECT_EQ(st, Status::Ok());

  DenseVarNames dvn;
  st = server.StreamingDenseVarName(0, &dvn);
  EXPECT_EQ(st, Status::Ok());

  DenseVarNames dvn2;
  dvn2.names.push_back("apple");
  dvn2.names.push_back("banana");
  DenseVarValues dvv;
  server.GatherStreamingDenseVar(0, dvn2, &dvv);
  EXPECT_EQ(st, Status::Ok());

  std::string ver("inc-99");
  st = server.TriggerStreamingSparse(0, 10, ver);
  EXPECT_NE(st, Status::Ok());

  st = server.TriggerStreamingHash(0, 10, ver);
  EXPECT_NE(st, Status::Ok());

  EXPECT_TRUE(server.RegisterUdfChain(kUnusedVersion, BuildUdfChainRegister()).IsOk());
  EXPECT_TRUE(server.RegisterUdfChain(kUnusedVersion, BuildUdfChainRegister()).IsOk());
  UdfContext ctx1;
  EXPECT_TRUE(server.RunUdfChain(kUnusedVersion, 100, "^var", Inputs(), &ctx1).IsOk());
  ctx1.GetStorageManager()->Set("var", []{ return new Variable(new Tensor(DataType::kInt8, TensorShape({4, 8}), new ConstantInitializer(1)), nullptr, "");});
  UdfContext ctx2;
  EXPECT_TRUE(server.RunUdfChain(kUnusedVersion, 100, "var", Inputs(), &ctx2).IsOk());
  UdfContext ctx3;
  EXPECT_FALSE(server.RunUdfChain(kUnusedVersion, 100, "var1", Inputs(), &ctx3).IsOk());
  EXPECT_FALSE(server.RunUdfChain(kUnusedVersion, 200, "var", Inputs(), &ctx3).IsOk());

  EXPECT_EQ(1u, ctx1.Outputs().size());
  EXPECT_EQ(10, dynamic_cast<WrapperData<int>*>(ctx1.Outputs()[0])->Internal());

  EXPECT_EQ(1u, ctx2.Outputs().size());
  EXPECT_EQ(40, dynamic_cast<WrapperData<int>*>(ctx2.Outputs()[0])->Internal());
}

TEST(LocalServerTest, LocalServer) {
  std::unique_ptr<LocalServer> svr(new LocalServer("./"));
  EXPECT_NE(svr, nullptr);

  Status st = svr->Save("123");
  EXPECT_EQ(st, Status::Ok());
  st = svr->Restore("123");
  EXPECT_EQ(st, Status::Ok());
}
