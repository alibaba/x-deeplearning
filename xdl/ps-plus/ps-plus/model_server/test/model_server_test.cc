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
#include "ps-plus/model_server/model_server.h"

using ps::Tensor;
using ps::Status;
using ps::DataType;
using ps::TensorShape;
using ps::modelserver::ModelServer;
using ps::modelserver::BackwardCache;
using ps::modelserver::BackwardRegistry;
using ps::modelserver::ForwardCache;
using ps::modelserver::ForwardRegistry;


TEST(ModelServerTest, ModelServer) {
  ForwardCache::Callback fcallback = [](Status st, Tensor tensor) {
    st = Status::Ok();
  };
  ForwardCache::ForwardRun frun = [](Tensor tensor, ForwardCache::Callback callback) {
    Status st; callback(st, tensor); ASSERT_EQ(st, Status::Ok());
  };

  BackwardCache::Callback bcallback = [](Status st) {
    st = Status::Ok();
  };
  BackwardCache::BackwardRun brun = [](Tensor tensor1, Tensor tensor2, BackwardCache::Callback callback) {
    Status st; callback(st); ASSERT_EQ(st, Status::Ok());
  };

  auto ms = new ModelServer(frun, brun, "name=no_cache", "name=no_cache");
  ASSERT_NE(ms, nullptr);
  Status st = ms->Init();
  ASSERT_EQ(st, Status::Ok());

  ms->Flush([](Status st) {});

  Tensor tensor;
  ms->RequestForward(tensor, fcallback);

  Tensor grad;
  ms->RequestBackward(tensor, grad, [](Status st){});

  delete ms;
}
