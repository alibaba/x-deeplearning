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
#include "ps-plus/common/tensor.h"
#include "ps-plus/model_server/backward.h"
#include "ps-plus/common/initializer/constant_initializer.h"

using ps::Tensor;
using ps::Status;
using ps::DataType;
using ps::TensorShape;
using ps::modelserver::BackwardCache;
using ps::modelserver::BackwardRegistry;
using ps::initializer::ConstantInitializer;

TEST(BackwardNoCacheTest, BackwardNoCache) {
  auto factory = ps::GetPlugin<BackwardRegistry>("no_cache");
  ASSERT_NE(factory, nullptr);
  Tensor tensor;
  BackwardCache::Callback callback = [&tensor](Status st) {
    st = Status::Ok();
  };
  BackwardCache::BackwardRun run = [&tensor, &callback](Tensor tensor1, Tensor tensor2, BackwardCache::Callback callback) {
    Status st; callback(st); ASSERT_EQ(st, Status::Ok());
  };

  auto fnc = factory->Create();
  std::unordered_map<std::string, std::string> hmap;
  hmap["a"] = "apple";
  Status st = fnc->Init(run, hmap);
  ASSERT_EQ(st, Status::Ok());

  Tensor grad;
  fnc->Calc(tensor, grad, callback);

  st = fnc->Flush();
  ASSERT_EQ(st, Status::Ok());
}

TEST(BackwardUniqueCacheTest, BackwardUniqueCache) {
  auto factory = ps::GetPlugin<BackwardRegistry>("unique_cache");
  ASSERT_NE(factory, nullptr);
  {
    Tensor tensor(DataType::kInt64, TensorShape({1024, 1024}), new ConstantInitializer(0));
    BackwardCache::Callback callback = [&tensor](Status st) {
      st = Status::Ok();
    };
    BackwardCache::BackwardRun run = [&tensor, &callback](Tensor tensor1, Tensor tensor2, BackwardCache::Callback callback) {
      Status st; callback(st); ASSERT_EQ(st, Status::Ok());
    };

    auto fnc = factory->Create();
    std::unordered_map<std::string, std::string> hmap;
    hmap["a"] = "apple";
    hmap["window_size"] = 1;
    Status st = fnc->Init(run, hmap);
    ASSERT_EQ(st, Status::Ok());

    Tensor grad(DataType::kInt64, TensorShape({1024, 1024}), new ConstantInitializer(1));
    fnc->Calc(tensor, grad, callback);

    st = fnc->Flush();
    ASSERT_EQ(st, Status::Ok());
  }

  {
    Tensor tensor(DataType::kInt16, TensorShape({1024, 1024}), new ConstantInitializer(0));
    BackwardCache::Callback callback = [](Status st) {
      st = Status::Ok();
    };

    auto fnc = factory->Create();
    Tensor grad(DataType::kInt64, TensorShape({1024}), new ConstantInitializer(1));;
    fnc->Calc(tensor, grad, callback);
  }

  {
    Tensor tensor(DataType::kInt64, TensorShape({1024}), new ConstantInitializer(0));
    BackwardCache::Callback callback = [](Status st) {
      st = Status::Ok();
    };
    BackwardCache::BackwardRun run = [](Tensor tensor1, Tensor tensor2, BackwardCache::Callback callback) {
      Status st; callback(st); ASSERT_EQ(st, Status::Ok());
    };

    auto fnc = factory->Create();
    std::unordered_map<std::string, std::string> hmap;
    hmap["a"] = "apple";
    hmap["window_size"] = 1;
    Status st = fnc->Init(run, hmap);
    ASSERT_EQ(st, Status::Ok());

    Tensor grad(DataType::kInt64, TensorShape({1024}), new ConstantInitializer(1));;
    fnc->Calc(tensor, grad, callback);

    Tensor grad1(DataType::kInt64, TensorShape({1024, 128}), new ConstantInitializer(1));;
    fnc->Calc(tensor, grad1, callback);
  }

  {
    Tensor tensor(DataType::kInt64, TensorShape({1024}), new ConstantInitializer(0));
    BackwardCache::Callback callback = [](Status st) {
      st = Status::Ok();
    };
    BackwardCache::BackwardRun run = [](Tensor tensor1, Tensor tensor2, BackwardCache::Callback callback) {
      Status st; callback(st); ASSERT_EQ(st, Status::Ok());
    };

    auto fnc = factory->Create();
    std::unordered_map<std::string, std::string> hmap;
    hmap["a"] = "apple";
    hmap["window_size"] = "1";
    Status st = fnc->Init(run, hmap);
    ASSERT_EQ(st, Status::Ok());

    Tensor grad(DataType::kInt64, TensorShape({1024}), new ConstantInitializer(1));;
    fnc->Calc(tensor, grad, callback);

    Tensor grad1(DataType::kInt64, TensorShape({1024, 128}), new ConstantInitializer(1));;
    fnc->Calc(tensor, grad1, callback);
  }

}
