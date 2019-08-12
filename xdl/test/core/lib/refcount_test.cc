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
#include "xdl/core/lib/refcount.h"

using xdl::RefCounted;
using xdl::RefCountedPtr;

namespace {
class MockRef : public RefCounted {
 public:
  explicit MockRef(int x = 1) : RefCounted(x) { refX++; }
  ~MockRef() { refX--; }
  static int refX;
};

int MockRef::refX = 0;

}  // namespace

TEST(RefCountedTest, RefCounted) {
  MockRef::refX = 0;
  MockRef* a = new MockRef();
  MockRef* b = new MockRef(2);
  ASSERT_EQ(2, MockRef::refX);
  a->UnRef();
  b->UnRef();
  ASSERT_EQ(1, MockRef::refX);
  b->UnRef();
  ASSERT_EQ(0, MockRef::refX);
}

TEST(RefCountedTest, RefCountedPtr) {
  MockRef::refX = 0;
  {
    RefCountedPtr<MockRef> ptr(new MockRef);
    ASSERT_EQ(1, MockRef::refX);
    ptr->UnRef();
  }
  ASSERT_EQ(0, MockRef::refX);
  RefCountedPtr<MockRef> ptr2;
  ASSERT_EQ(0, MockRef::refX);
  {
    RefCountedPtr<MockRef> ptr3 = RefCountedPtr<MockRef>::Create();
    ASSERT_EQ(1, MockRef::refX);
  }
  ASSERT_EQ(0, MockRef::refX);
  {
    RefCountedPtr<MockRef> ptr4 = RefCountedPtr<MockRef>::Create();
    ASSERT_EQ(1, MockRef::refX);
    {
      RefCountedPtr<MockRef> ptr5 = ptr4;
      ASSERT_EQ(1, MockRef::refX);
    }
    {
      RefCountedPtr<MockRef> ptr6;
      ptr6 = ptr4; /* Test Assignment operator */
      ASSERT_EQ(1, MockRef::refX);
    }
    ASSERT_EQ(1, MockRef::refX);
    ptr4->UnRef();
    ASSERT_EQ(0, MockRef::refX);
  }
  ASSERT_EQ(0, MockRef::refX);
}
