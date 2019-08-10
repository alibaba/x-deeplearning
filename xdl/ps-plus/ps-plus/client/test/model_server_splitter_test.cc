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
#include "ps-plus/common/initializer.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/client/model_server_splitter.h"

using ps::Status;
using ps::Tensor;
using ps::TensorShape;
using ps::initializer::ConstantInitializer;
using ps::DataType;
using ps::client::ModelServerSplitter;

TEST(ModelServerSplitterTest, ModelServerSplitter) {
  std::unique_ptr<ModelServerSplitter> mss(new ModelServerSplitter());
  EXPECT_NE(mss, nullptr);

  {
    Tensor tensor(DataType::kInt16, TensorShape({32, 1024}), new ConstantInitializer(0));
    Status st = mss->Init(1, tensor);
    EXPECT_NE(st, Status::Ok());
  }

  {
    Tensor tensor(DataType::kInt16, TensorShape({1024}), new ConstantInitializer(0));
    Status st = mss->Init(1, tensor);
    EXPECT_NE(st, Status::Ok());
  }

  {
    Tensor tensor(DataType::kInt64, TensorShape({1024}), new ConstantInitializer(0));
    Status st = mss->Init(2, tensor);
    EXPECT_EQ(st, Status::Ok());

    Tensor tensor2(DataType::kInt64, TensorShape({1024, 512}), new ConstantInitializer(0));
    std::vector<Tensor> vec;
    st = mss->Split(tensor2, &vec);
    EXPECT_EQ(st, Status::Ok());
    EXPECT_EQ(vec.size(), 2);
  }

  {
    Tensor tensor(DataType::kInt64, TensorShape({1024}), new ConstantInitializer(0));
    Status st = mss->Init(2, tensor);
    EXPECT_EQ(st, Status::Ok());

    Tensor tensor2(DataType::kInt64, TensorShape({1024, 512}), new ConstantInitializer(0));
    Tensor rst(DataType::kInt64, TensorShape({1024, 512}), new ConstantInitializer(0), Tensor::TType::kSegment, false);;
    st = mss->Combine(0, tensor2, &rst);
    EXPECT_EQ(st, Status::Ok());
  }

}
