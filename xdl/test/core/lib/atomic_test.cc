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
#include "xdl/core/lib/atomic.h"

using namespace xdl::common;

TEST(AtomicTEST, Atomic) {
  int32_t value = 2;
  int32_t new_val = cpu_atomic_add(1, &value);
  ASSERT_EQ(new_val, 2);
  ASSERT_EQ(value, 3);
}
