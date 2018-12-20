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
#include "xdl/core/framework/graph_def.h"
#include "xdl/core/framework/grappler.h"

using xdl::AttrValue;
using xdl::NodeDef;
using xdl::DataType;
using xdl::OutputSpec;
using xdl::TensorShape;
using xdl::Status;
using xdl::GraphDef;
using xdl::Grappler;
using xdl::GrapplerRegistry;

namespace {

GraphDef CreateDef() {
  NodeDef a, b, c;

  a.name = "a";
  a.op = "MockConstant";
  a.device.device_name = "CPU";
  a.attr["dtype"].attr_type = AttrValue::kDataType;
  a.attr["dtype"].type = DataType::kInt64;
  a.attr["shape"].attr_type = AttrValue::kTensorShape;
  a.attr["shape"].shape = TensorShape({10, 20});

  b.name = "b";
  b.op = "MockConstant";
  b.device.device_name = "CPU";
  b.attr["dtype"].attr_type = AttrValue::kDataType;
  b.attr["dtype"].type = DataType::kInt64;
  b.attr["shape"].attr_type = AttrValue::kTensorShape;
  b.attr["shape"].shape = TensorShape({10, 20});
  b.attr["value"].attr_type = AttrValue::kInt;
  b.attr["value"].i = 20;

  c.name = "c";
  c.op = "MockMulAdd";
  c.device.device_name = "CPU";
  c.input.push_back("a:0");
  c.input.push_back("b:0");
  c.attr["dtype"].attr_type = AttrValue::kDataType;
  c.attr["dtype"].type = DataType::kInt64;
  c.attr["add"].attr_type = AttrValue::kInt;
  c.attr["add"].i = 30;

  GraphDef ret;
  ret.node.push_back(a);
  ret.node.push_back(b);
  ret.node.push_back(c);
  ret.hash = 123;
  return ret;
}

OutputSpec CreateOutput() {
  OutputSpec ret;
  ret.output.push_back("c:0");
  ret.output_device.device_name = "CPU";
  return ret;
}

}

TEST(GrapplerTest, Grappler) {
  xdl::GraphDef def = CreateDef();
  xdl::OutputSpec output = CreateOutput();

  GrapplerRegistry *registry = GrapplerRegistry::Get();
  ASSERT_TRUE(registry != nullptr);

  Status st = registry->Process(&def, &output);
  ASSERT_EQ(st, Status::Ok());
}
