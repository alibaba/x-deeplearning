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
#include "ps-plus/server/streaming_model_utils.h"
#include "ps-plus/common/initializer/none_initializer.h"
#include "ps-plus/common/initializer/constant_initializer.h"

#include <thread>
#include <future>

using ps::Tensor;
using ps::Initializer;
using ps::DataType;
using ps::TensorShape;
using ps::Status;
using ps::server::StreamingModelUtils;
using ps::initializer::NoneInitializer;
using ps::initializer::ConstantInitializer;

TEST(StreamingModelUtilsTest, StreamingModelUtilsTest) {
  std::vector<std::thread> threads;
  StreamingModelUtils utils;
  std::vector<Status> results(30);
  std::vector<std::promise<int>> ready(10);
  std::vector<std::promise<int>*> destory(10);
  for (int i = 0; i < 10; i++) {
    threads.emplace_back([&, i]{
      Tensor x(DataType::kInt64, TensorShape({1}), new ConstantInitializer(i));
      Tensor y(DataType::kInt64, TensorShape({1, 2}), new ConstantInitializer(i / 3));
      Status st1 = utils.WriteDense(std::to_string(i));
      Status st2 = utils.WriteSparse("sparse" + std::to_string(i / 5), x);
      Status st3 = utils.WriteHash("hash" + std::to_string(i / 5), y);
      results[i * 3] = st1;
      results[i * 3 + 1] = st2;
      results[i * 3 + 2] = st3;
      ready[i].set_value(1);
      std::promise<int> d;
      destory[i] = &d;
      d.get_future().wait();
    });
  }
  for (int i = 0; i < 10; i++) {
    ready[i].get_future().wait();
  }
  for (int i = 0; i < 30; i++) {
    EXPECT_EQ("", results[i].Msg());
  }
  std::unordered_map<std::string, StreamingModelUtils::DenseLog> dense;
  std::unordered_map<std::string, StreamingModelUtils::SparseLog> sparse;
  std::unordered_map<std::string, StreamingModelUtils::HashLog> hash;

  EXPECT_TRUE(utils.GetDense(&dense).IsOk());
  EXPECT_TRUE(utils.GetSparse(&sparse).IsOk());
  EXPECT_TRUE(utils.GetHash(&hash).IsOk());
  EXPECT_EQ(10u, dense.size());
  EXPECT_EQ(2u, sparse.size());
  EXPECT_EQ(2u, hash.size());
  EXPECT_EQ(5u, sparse["sparse0"].write_ids.size());
  EXPECT_EQ(5u, sparse["sparse1"].write_ids.size());
  EXPECT_EQ(2u, hash["hash0"].write_ids.size());
  EXPECT_EQ(3u, hash["hash1"].write_ids.size());

  EXPECT_TRUE(utils.GetDense(&dense).IsOk());
  EXPECT_TRUE(utils.GetSparse(&sparse).IsOk());
  EXPECT_TRUE(utils.GetHash(&hash).IsOk());
  EXPECT_EQ(0u, dense.size());
  EXPECT_EQ(0u, sparse.size());
  EXPECT_EQ(0u, hash.size());

  EXPECT_TRUE(utils.GetDense(&dense).IsOk());
  EXPECT_TRUE(utils.GetSparse(&sparse).IsOk());
  EXPECT_TRUE(utils.GetHash(&hash).IsOk());
  EXPECT_EQ(0u, dense.size());
  EXPECT_EQ(0u, sparse.size());
  EXPECT_EQ(0u, hash.size());

  for (int i = 0; i < 10; i++) {
    destory[i]->set_value(1);
  }
  for (int i = 0; i < 10; i++) {
    threads[i].join();
  }
}

