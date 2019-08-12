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
#include "xdl/core/framework/tensor.h"

using xdl::Tensor;
using xdl::Device;
using xdl::Allocator;
using xdl::TensorShape;
using xdl::DataType;
using xdl::ThreadPool;
using xdl::OpKernelBase;
using xdl::OpKernelContext;

namespace {

class MockAllocator : public Allocator {
 public:
  void* Allocate(size_t size) override {
    refX++;
    return reinterpret_cast<void*>(&buf);
  }
  void Deallocate(void* b) override {
    (void)b;
    refX--;
  }
  static int64_t buf;
  static int refX;
};

class MockDevice : public Device {
 public:
  MockDevice() : Device(new MockAllocator) {}
  std::string DeviceType() override {
    return "MockDevice";
  }
  static std::string name;
};

int64_t MockAllocator::buf = 0;
int MockAllocator::refX = 0;

}  // namespace

TEST(TensorTest, Tensor) {
  std::unique_ptr<MockDevice> device(new MockDevice);
  {
    Tensor t0;
    {
      Tensor t1(device.get(), TensorShape({100, 10}), DataType::kInt64);
      ASSERT_EQ(1, MockAllocator::refX);
      ASSERT_EQ(&MockAllocator::buf, t1.Raw<int64_t>());
      Tensor t2 = t1;
      ASSERT_EQ(1, MockAllocator::refX);
    }
    ASSERT_EQ(0, MockAllocator::refX);
    {
      Tensor t3(device.get(), TensorShape({100, 10}), DataType::kInt64);
      ASSERT_EQ(1, MockAllocator::refX);
      Tensor t4(device.get(), TensorShape({100, 10}), DataType::kInt64);
      ASSERT_EQ(2, MockAllocator::refX);
    }
    ASSERT_EQ(0, MockAllocator::refX);
  }
}

