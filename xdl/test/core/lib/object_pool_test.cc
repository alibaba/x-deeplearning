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

/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "xdl/core/lib/object_pool.h"
#include "gtest/gtest.h"

namespace xdl {

TEST(TestObjectPool, All) {
  ObjectPool<int> object_pool;
  
  int* a1 = object_pool.Acquire();
  *a1 = 1;
  object_pool.Release(a1);
  a1 = object_pool.Acquire();
  EXPECT_EQ(*a1, 1);
  EXPECT_EQ(object_pool.Size(), 0);

  int* b1 = object_pool.Acquire();
  *b1 = 2;
  std::vector<int*> objects = { a1, b1 };
  object_pool.Release(objects);
  EXPECT_EQ(object_pool.Size(), 2);

  b1 = object_pool.Acquire();
  EXPECT_EQ(*b1, 2);
}

TEST(TestMultiObjectPool, All) {
  MultiObjectPool<int> mop;

  int* a1 = mop.Acquire("mama");
  *a1 = 1;
  mop.Release("mama", a1);

  a1 = mop.Acquire("baba");
  *a1 = 2;
  mop.Release("baba", a1);

  a1 = mop.Acquire("mama");
  EXPECT_EQ(*a1, 1);
  a1 = mop.Acquire("baba");
  EXPECT_EQ(*a1, 2);
}

}  // namespace xdl
