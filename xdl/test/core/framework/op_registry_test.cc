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
#include "xdl/core/framework/op_registry.h"

using xdl::OpKernelBase;
using xdl::OpKernelContext;
using xdl::NodeDef;
using xdl::OpRegistry;
using xdl::AttrValue;

namespace {

template<int id>
class MockOpKernel : public OpKernelBase {
 public:
  void Launch(OpKernelContext* ctx) override {
  }
  static bool IsInstance(OpKernelBase* kernel) {
    return dynamic_cast<MockOpKernel*>(kernel) != nullptr;
  }
};

}  // namespace

XDL_REGISTER_KERNEL(OpRegistryMockOp, MockOpKernel<0>).Priority(-1);
XDL_REGISTER_KERNEL(OpRegistryMockOp, MockOpKernel<1>).Device("X");
XDL_REGISTER_KERNEL(OpRegistryMockOp, MockOpKernel<2>).Device("Y");
XDL_REGISTER_KERNEL(OpRegistryMockOp, MockOpKernel<3>).AttrInt("I", 1);

TEST(OpRegistryTest, OpRegistry) {
  NodeDef node_a;
  node_a.name = "A";
  node_a.op = "OpRegistryMockOp";
  NodeDef node_b;
  node_b.name = "A";
  node_b.op = "OpRegistryMockOp";
  node_b.attr["I"].attr_type = AttrValue::kInt;
  node_b.attr["I"].i = 1;
  NodeDef node_c;
  node_c.name = "A";
  node_c.op = "OpRegistryMockOp";
  node_c.attr["I"].attr_type = AttrValue::kInt;
  node_c.attr["I"].i = 2;
  OpKernelBase* kernel;
  ASSERT_TRUE(OpRegistry::Get()
      ->CreateKernel(node_a, "X", &kernel).IsOk());
  ASSERT_TRUE(MockOpKernel<1>::IsInstance(kernel));
  ASSERT_TRUE(OpRegistry::Get()
      ->CreateKernel(node_a, "Y", &kernel).IsOk());
  ASSERT_TRUE(MockOpKernel<2>::IsInstance(kernel));
  ASSERT_TRUE(OpRegistry::Get()
      ->CreateKernel(node_a, "Z", &kernel).IsOk());
  ASSERT_TRUE(MockOpKernel<0>::IsInstance(kernel));
  ASSERT_TRUE(OpRegistry::Get()
      ->CreateKernel(node_a, "Z", &kernel).IsOk());
  ASSERT_TRUE(MockOpKernel<0>::IsInstance(kernel));
  ASSERT_TRUE(OpRegistry::Get()
      ->CreateKernel(node_b, "Z", &kernel).IsOk());
  ASSERT_TRUE(MockOpKernel<3>::IsInstance(kernel));
  ASSERT_TRUE(OpRegistry::Get()
      ->CreateKernel(node_c, "Z", &kernel).IsOk());
  ASSERT_TRUE(MockOpKernel<0>::IsInstance(kernel));
}

