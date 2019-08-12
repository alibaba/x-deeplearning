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
#include "ps-plus/common/bloom_filter.h"
#include <vector>

TEST(BloomFilter, CountingBloomFilter) {
  double fpp = 0.001;
  uint64_t elem_size = 1000;
  ps::CountingBloomFilter<uint8_t> cbf(fpp, elem_size);
  ASSERT_EQ(cbf.hash_function_number(), 10);
  ASSERT_GE(cbf.bucket_size(), 14378);
  
  std::vector<uint64_t> keys = { 1ul, 2ul, 3ul, 4ul, 1ul, 2ul, 2ul };
  int sz1 = sizeof(uint64_t);
  int sz2 = sz1 * 2;
  ASSERT_TRUE(cbf.Exists(nullptr, 0, 0));
  ASSERT_FALSE(cbf.Exists(nullptr, 0, 1));
  ASSERT_TRUE(cbf.Exists(&keys[0], sz1, 0));
  cbf.Insert(&keys[0], sz1);
  ASSERT_TRUE(cbf.Exists(&keys[0], sz1, 1));
  // must be false
  ASSERT_FALSE(cbf.Exists(&keys[0], sz1, 2));
  // may be false
  EXPECT_FALSE(cbf.Exists(&keys[1], sz1, 1));
  // may be false 
  EXPECT_FALSE(cbf.Exists(&keys[0], sz2, 1));
  cbf.Insert(&keys[1], sz1);
  cbf.Insert(&keys[2], sz1);
  cbf.Insert(&keys[3], sz1);
  cbf.Insert(&keys[4], sz1);
  ASSERT_TRUE(cbf.Exists(&keys[4], sz1, 2));
  cbf.Insert(&keys[5], sz1);
  cbf.Insert(&keys[6], sz1);
  ASSERT_TRUE(cbf.Exists(&keys[1], sz1, 3));
}

TEST(BloomFilter, InsertedLookup) {
  double fpp = 0.001;
  uint64_t elem_size = 1000;
  ps::CountingBloomFilter<uint8_t> cbf(fpp, elem_size);
  uint64_t key = 123;
  for (int i = 0; i < 300; ++i) {
    bool res = cbf.InsertedLookup(&key, sizeof(key), 255);
    if (i >= 255) ASSERT_TRUE(res);
  }
  ASSERT_FALSE(cbf.InsertedLookup(&key, sizeof(key), 256));
}

