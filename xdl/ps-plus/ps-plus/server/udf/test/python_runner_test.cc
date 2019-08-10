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
#include "ps-plus/server/udf/python_runner.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include <iostream>

using ps::PythonRunner;
using ps::DataType;
using ps::TensorShape;
using ps::Tensor;
using ps::Status;
using ps::initializer::ConstantInitializer;

TEST(PythonRunner, PythonRunner) {
  std::string func_def = "import numpy\ndef X():\n print 'Hello World';\n return 1";
  std::string func_name = "X";
  PythonRunner runner;

  Status status = runner.Init(func_def, func_name);
  EXPECT_TRUE(status.IsOk());
  PythonRunner::NumpyArray arr;
  status = runner.Run({}, &arr);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(1, ((int*)arr.data)[0]);
}

TEST(PythonRunner, Add) {
  std::string func_def = "import numpy\ndef Y(a, b):\n return a + b";
  std::string func_name = "Y";
  PythonRunner runner;
  Status status = runner.Init(func_def, func_name);
  EXPECT_TRUE(status.IsOk());
  Tensor t1(DataType::kFloat, TensorShape({3, 8}), new ConstantInitializer(1), true, 2);
  Tensor t2(DataType::kFloat, TensorShape({3, 8}), new ConstantInitializer(2), true, 2);
  PythonRunner::NumpyArray r1, r2, r3;
  status = PythonRunner::ParseSubTensor(t1, 1, 3, &r1);
  EXPECT_TRUE(status.IsOk());
  status = PythonRunner::ParseSubTensor(t2, 1, 3, &r2);
  EXPECT_TRUE(status.IsOk());
  status = runner.Run({r1, r2}, &r3);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(3, ((float*)r3.data)[0]);
  EXPECT_EQ(1, r1.shape[0]);
  EXPECT_EQ(8, r1.shape[1]);
  EXPECT_EQ(1, r2.shape[0]);
  EXPECT_EQ(8, r2.shape[1]);
  EXPECT_EQ(1, r3.shape[0]);
  EXPECT_EQ(8, r3.shape[1]);
}

TEST(PythonRunner, Swap) {
  std::string func_def = "import numpy\ndef Z(a, b):\n return b, a";
  std::string func_name = "Z";
  PythonRunner runner;
  Status status = runner.Init(func_def, func_name);
  EXPECT_TRUE(status.IsOk());
  Tensor t1(DataType::kFloat, TensorShape({3, 8}), new ConstantInitializer(1));
  Tensor t2(DataType::kFloat, TensorShape({3, 8}), new ConstantInitializer(2));
  PythonRunner::NumpyArray r1, r2, r3, r4;
  status = PythonRunner::ParseTensor(t1, &r1);
  EXPECT_TRUE(status.IsOk());
  status = PythonRunner::ParseTensor(t2, &r2);
  EXPECT_TRUE(status.IsOk());
  status = runner.Run({r1, r2}, &r3, &r4);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(2, ((float*)r3.data)[0]);
  EXPECT_EQ(1, ((float*)r4.data)[0]);
  EXPECT_EQ(3, r1.shape[0]);
  EXPECT_EQ(8, r1.shape[1]);
  EXPECT_EQ(3, r2.shape[0]);
  EXPECT_EQ(8, r2.shape[1]);
  EXPECT_EQ(3, r3.shape[0]);
  EXPECT_EQ(8, r3.shape[1]);
  EXPECT_EQ(3, r4.shape[0]);
  EXPECT_EQ(8, r4.shape[1]);
}
