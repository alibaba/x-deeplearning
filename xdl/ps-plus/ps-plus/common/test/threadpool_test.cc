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
#include "ps-plus/common/thread_pool.h"

using ps::ThreadPool;

TEST(ThreadPoolTest, ThreadPool) {
  const size_t size = 3ul << 30; /* 3G */
  char *dest = (char*)malloc(size); /* 1 MegaBytes */
  char *src = (char*)malloc(size);
  ASSERT_NE(dest, nullptr);
  ASSERT_NE(src, nullptr);
  memset(src, 'c', size);

  ps::QuickMemcpy(dest, src, size);

  ASSERT_EQ(dest[0], 'c');
  ASSERT_EQ(dest[size - 1], 'c');

  free(dest);
  free(src);
}
