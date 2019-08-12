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
#include "ps-plus/model_server/forward.h"
#include "ps-plus/common/initializer/constant_initializer.h"

using ps::Tensor;
using ps::Status;
using ps::DataType;
using ps::TensorShape;
using ps::modelserver::ForwardCache;
using ps::modelserver::ForwardRegistry;
using ps::initializer::ConstantInitializer;

TEST(ForwardNoCacheTest, ForwardNoCache) {
  auto factory = ps::GetPlugin<ForwardRegistry>("no_cache");
  ASSERT_NE(factory, nullptr);
  Tensor tensor;
  ForwardCache::Callback callback = [&tensor](Status st, Tensor tensor) {
    st = Status::Ok();
  };
  ForwardCache::ForwardRun run = [&tensor, &callback](Tensor tensor, ForwardCache::Callback callback) {
    Status st; callback(st, tensor); ASSERT_EQ(st, Status::Ok());
  };

  auto fnc = factory->Create();
  std::unordered_map<std::string, std::string> hmap;
  hmap["a"] = "apple";
  Status st = fnc->Init(run, hmap);
  ASSERT_EQ(st, Status::Ok());

  fnc->Calc(tensor, callback);

  st = fnc->Flush();
  ASSERT_EQ(st, Status::Ok());
}

TEST(ForwardSimpleCacheTest, ForwardSimpleCache) {
  auto factory = ps::GetPlugin<ForwardRegistry>("simple_cache");
  ASSERT_NE(factory, nullptr);
  {
    Tensor tensor(DataType::kInt64, TensorShape({1, 1024}), new ConstantInitializer(0), Tensor::TType::kContinuous, false);
    ForwardCache::Callback callback = [&tensor](Status st, Tensor tensor) {
      st = Status::Ok();
    };
    ForwardCache::ForwardRun run = [](Tensor tensor, ForwardCache::Callback callback) {
      Status st;
      callback(st, tensor);
      ASSERT_EQ(st, Status::Ok());
    };

    auto fnc = factory->Create();
    std::unordered_map<std::string, std::string> hmap;
    hmap["a"] = "apple";
    Status st = fnc->Init(run, hmap);
    ASSERT_EQ(st, Status::Ok());

    fnc->Calc(tensor, callback);

    st = fnc->Flush();
    ASSERT_EQ(st, Status::Ok());
  }

  {
    Tensor tensor(DataType::kInt64, TensorShape({128, 128}), new ConstantInitializer(1));
    ForwardCache::Callback callback = [&tensor](Status st, Tensor tensor) {
      st = Status::Ok();
    };
    ForwardCache::ForwardRun run = [&tensor, &callback](Tensor tensor, ForwardCache::Callback callback) {
      Status st; callback(st, tensor); ASSERT_EQ(st, Status::Ok());
    };

    auto fnc = factory->Create();
    std::unordered_map<std::string, std::string> hmap;
    hmap["a"] = "apple";
    Status st = fnc->Init(run, hmap);
    ASSERT_EQ(st, Status::Ok());

    fnc->Calc(tensor, callback);

    st = fnc->Flush();
    ASSERT_EQ(st, Status::Ok());
  }

}

TEST(ForwardUniqueCacheTest, ForwardUniqueCache) {
  auto factory = ps::GetPlugin<ForwardRegistry>("unique_cache");
  ASSERT_NE(factory, nullptr);
  {
    Tensor tensor(DataType::kInt64, TensorShape({1024, 1024}), new ConstantInitializer(0));
    ForwardCache::Callback callback = [&tensor](Status st, Tensor tensor) {
      st = Status::Ok();
    };
    ForwardCache::ForwardRun run = [](Tensor tensor, ForwardCache::Callback callback) {
      Status st; callback(st, tensor); ASSERT_EQ(st, Status::Ok());
    };

    auto fnc = factory->Create();
    std::unordered_map<std::string, std::string> hmap;
    hmap["a"] = "apple";
    hmap["window_size"] = "1";
    Status st = fnc->Init(run, hmap);
    ASSERT_EQ(st, Status::Ok());

    fnc->Calc(tensor, callback);

    st = fnc->Flush();
    ASSERT_EQ(st, Status::Ok());
  }

  {
    Tensor tensor(DataType::kInt16, TensorShape({1024, 1024}), new ConstantInitializer(0));
    ForwardCache::Callback callback = [&tensor](Status st, Tensor tensor) {
      st = Status::Ok();
    };

    auto fnc = factory->Create();
    fnc->Calc(tensor, callback);
  }

  {
    Tensor tensor(DataType::kInt64, TensorShape({1024}), new ConstantInitializer(0));
    ForwardCache::Callback callback = [&tensor](Status st, Tensor tensor) {
      st = Status::Ok();
    };
    ForwardCache::ForwardRun run = [](Tensor tensor, ForwardCache::Callback callback) {
      Status st; callback(st, tensor); ASSERT_EQ(st, Status::Ok());
    };

    auto fnc = factory->Create();
    std::unordered_map<std::string, std::string> hmap;
    hmap["a"] = "apple";
    hmap["window_size"] = "1";
    Status st = fnc->Init(run, hmap);
    ASSERT_EQ(st, Status::Ok());

    fnc->Calc(tensor, callback);
  }

}
