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

#include "ps-plus/profiler/profiler.h"
#include "ps-plus/server/server.h"
#include "ps-plus/common/initializer/constant_initializer.h"

using ps::server::Server;
using ps::TensorShape;
using ps::DataType;
using ps::initializer::ConstantInitializer;
using ps::Initializer;
using ps::Data;
using ps::WrapperData;
using ps::Tensor;
using ps::UdfChainRegister;
using ps::kUnusedVersion;
using ps::server::UdfContext;
using ps::server::StreamingModelArgs;

static std::unique_ptr<Server> server_;

static UdfChainRegister DenseInit() {
  UdfChainRegister def;
  def.hash = 1;
  def.udfs.resize(2);
  def.udfs[0].udf_name = "";
  def.udfs[0].inputs = {};
  def.udfs[1].udf_name = "DenseVariableInitializer";
  def.udfs[1].inputs = {{0, 0}, {0, 1}, {0, 2}};
  def.outputs = {{1, -1}};
  return def;
}

static UdfChainRegister DensePull() {
  UdfChainRegister def;
  def.hash = 2;
  def.udfs.resize(2);
  def.udfs[0].udf_name = "";
  def.udfs[0].inputs = {};
  def.udfs[1].udf_name = "BuildDenseSlice";
  def.udfs[1].inputs = {};
  def.outputs = {{1, -1}};
  return def;
}

static UdfChainRegister DensePush() {
  UdfChainRegister def;
  def.hash = 3;
  def.udfs.resize(3);
  def.udfs[0].udf_name = "";
  def.udfs[0].inputs = {};
  def.udfs[1].udf_name = "BuildDenseSlice";
  def.udfs[1].inputs = {};
  def.udfs[2].udf_name = "MomentumUpdater";
  def.udfs[2].inputs = {{1, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 3}};
  def.outputs = {{2, -1}};
  return def;
}

PROFILE(dense_pull_push, 32, 1000).Init([](size_t threads){
  server_.reset(new Server(0, StreamingModelArgs()));
  server_->RegisterUdfChain(kUnusedVersion, DenseInit());
  server_->RegisterUdfChain(kUnusedVersion, DensePull());
  server_->RegisterUdfChain(kUnusedVersion, DensePush());
  std::vector<Data*> initializer_args;
  initializer_args.push_back(new WrapperData<DataType>(DataType::kFloat));
  initializer_args.push_back(new WrapperData<TensorShape>(std::vector<size_t>({1024})));
  initializer_args.push_back(new WrapperData<std::unique_ptr<Initializer>>(new ConstantInitializer(0)));
  UdfContext ctx;
  server_->RunUdfChain(kUnusedVersion, 1, "^var", initializer_args, &ctx);
}).TestCase([](size_t thread_id, bool run){
  if (run) {
    for (int i = 0; i < 10; i++) {
      UdfContext ctx1;
      server_->RunUdfChain(kUnusedVersion, 2, "var", std::vector<Data*>(), &ctx1);
      UdfContext ctx2;
      std::vector<Data*> push_args;
      push_args.push_back(new WrapperData<Tensor>(DataType::kFloat, TensorShape({1024}), new ConstantInitializer(1)));
      push_args.push_back(new WrapperData<double>(1));
      push_args.push_back(new WrapperData<double>(0.5));
      push_args.push_back(new WrapperData<bool>(false));
      server_->RunUdfChain(kUnusedVersion, 3, "var", push_args, &ctx2);
    }
  }
});
