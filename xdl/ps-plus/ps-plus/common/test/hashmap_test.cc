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

#include <iostream>
#include "gtest/gtest.h"
#include "ps-plus/common/hashmap.h"

using ps::HashMap;
using ps::HashMapItem;
using std::vector;

TEST(HashMapTest, Get) {
  HashMap hashmap(1280);
  int64_t keys[] = {1, 2, 3, 4}; 
  vector<int64_t> ids;
  vector<int64_t> reused_ids;
  int ret = hashmap.Get(keys, 2, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);   
  EXPECT_EQ(2u, ids.size());
  // NOTE: the latter part of keys would get id first
  EXPECT_EQ(0, ids[1]);
  EXPECT_EQ(1, ids[0]);
  EXPECT_EQ(0u, reused_ids.size());

  int64_t keys1[] = {1, 2, 3, 4, 13, 14};
  ret = hashmap.Get(keys1, 3, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);   
  EXPECT_EQ(3u, ids.size());
  // alloc new id
  EXPECT_EQ(2, ids[2]);
  // old id
  EXPECT_EQ(0, ids[1]);
  EXPECT_EQ(1, ids[0]);
  EXPECT_EQ(0u, reused_ids.size());
}

TEST(HashMapTest, GetWithVerChange) {
  HashMap hashmap(1280);
  hashmap.SetMaxCache(2);
  int64_t keys[] = {1, 2, 3, 4, 5, 6}; 
  vector<int64_t> ids;
  vector<int64_t> reused_ids;
  int ret = hashmap.Get(keys, 3, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);   
  EXPECT_EQ(3u, ids.size());
  // NOTE: the latter part of keys would get id first
  EXPECT_EQ(0, ids[2]);
  EXPECT_EQ(1, ids[1]);
  EXPECT_EQ(2, ids[0]);
  EXPECT_EQ(0u, reused_ids.size());

  int64_t keys1[] = {1, 2, 3, 4, 7, 8}; 
  ret = hashmap.Get(keys1, 3, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);   
  EXPECT_EQ(3u, ids.size());
  // <7, 8> new id:3
  EXPECT_EQ(3, ids[2]);
  EXPECT_EQ(1, ids[1]);
  EXPECT_EQ(2, ids[0]);
  EXPECT_EQ(0u, reused_ids.size());
} 

TEST(HashMapTest, Del) {
  HashMap hashmap(1280);
  int64_t keys[] = {1, 2, 3, 4};
  vector<int64_t> ids;
  vector<int64_t> reused_ids;
  int ret = hashmap.Get(keys, 2, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);  
  int64_t del_keys[] = {3, 4};
  ret = hashmap.Del(del_keys, 1, 2);
  ASSERT_EQ(0, ret);
  int64_t keys1[] = {1, 2, 5, 6};
  ret = hashmap.Get(keys1, 2, 2, &ids, &reused_ids);
  EXPECT_EQ(2u, ids.size());
  EXPECT_EQ(1, ids[0]);
  // reuse id:0 
  EXPECT_EQ(0, ids[1]);
  EXPECT_EQ(1u, reused_ids.size());
  EXPECT_EQ(0, reused_ids[0]);
  
  int64_t del_keys1[] = {1, 2};
  ret = hashmap.Del(del_keys1, 1, 2);
  ASSERT_EQ(0, ret);
  int64_t keys2[] = {1, 2, 3, 4, 5, 6, 7, 8};
  ret = hashmap.Get(keys2, 4, 2, &ids, &reused_ids);
  EXPECT_EQ(4u, ids.size());
  // <7, 8> reuse id:1
  EXPECT_EQ(1, ids[3]);
  // <5, 6> old id:0
  EXPECT_EQ(0, ids[2]);
  // <3, 4> new id:2
  EXPECT_EQ(2, ids[1]); 
  // <1, 2> new id:3
  EXPECT_EQ(3, ids[0]);
  EXPECT_EQ(1u, reused_ids.size());
  EXPECT_EQ(1, reused_ids[0]);

  int64_t del_keys2[] = {3, 4, 5, 6};
  ret = hashmap.Del(del_keys2, 2, 2);
  ASSERT_EQ(0, ret);
  int64_t keys3[] = {7, 8, 9, 10, 11, 12};
  ret = hashmap.Get(keys3, 3, 2, &ids, &reused_ids);
  EXPECT_EQ(3u, ids.size());
  // <11, 12> reuse id:2
  EXPECT_EQ(2, ids[2]);
  // <9, 10> reuse id:0
  EXPECT_EQ(0, ids[1]);
  // <7, 8> old id:1
  EXPECT_EQ(1, ids[0]); 
  EXPECT_EQ(2u, reused_ids.size());
  EXPECT_EQ(2, reused_ids[0]);
  EXPECT_EQ(0, reused_ids[1]);
}

TEST(HashMapTest, DelWithVerChange) {
  HashMap hashmap(1280);
  hashmap.SetMaxCache(2);
  int64_t keys[] = {1, 2, 3, 4, 5, 6, 7, 8}; 
  vector<int64_t> ids;
  vector<int64_t> reused_ids;
  // alloc id: 3->2->1->0
  int ret = hashmap.Get(keys, 4, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);   

  int64_t del_keys[] = {1, 2, 3, 4, 7, 8};
  // release id: 3->2->0
  ret = hashmap.Del(del_keys, 3, 2);
  ASSERT_EQ(0, ret);
 
  int64_t keys1[] = {5, 6, 3, 4, 9, 10};
  ret = hashmap.Get(keys1, 3, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);  
  EXPECT_EQ(3u, ids.size());
  // <9, 10> reuse id: 3
  EXPECT_EQ(3, ids[2]); 
  // <3, 4> reuse id: 2
  EXPECT_EQ(2, ids[1]); 
  // <5, 6> old id: 1
  EXPECT_EQ(1, ids[0]); 
}

TEST(HashMapTest, ExpandSpace) {
  HashMap hashmap(1);
  hashmap.SetMaxCache(100);
  int test_cnt = 140;
  int64_t keys[test_cnt*2];
  for (int i = 0; i < test_cnt; i++) {
    keys[i*2] = i*2;
    keys[i*2+1] = i*2+1;
  }
  vector<int64_t> ids;
  vector<int64_t> reused_ids;
  int ret = hashmap.Get(keys, test_cnt, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);
  EXPECT_EQ(test_cnt*1u, ids.size());
  for (int i = 0; i < test_cnt; i++) {
    EXPECT_EQ(i, ids[test_cnt - 1 - i]);
  }

  int64_t keys1[] = {0, 1, 2, 3, 278, 279, 300, 301};
  ret = hashmap.Get(keys1, 4, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);
  EXPECT_EQ(4u, ids.size());
  EXPECT_EQ(139, ids[0]);
  EXPECT_EQ(138, ids[1]);
  EXPECT_EQ(0, ids[2]);
  EXPECT_EQ(140, ids[3]);
}

TEST(HashMapTest, GetKeysAndSetKeys) {
  HashMap hashmap(1280);
  int64_t keys[] = {1, 2, 3, 4}; 
  vector<int64_t> ids;
  vector<int64_t> reused_ids;
  int ret = hashmap.Get(keys, 2, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);   
  EXPECT_EQ(2u, ids.size());
  // NOTE: the latter part of keys would get id first
  EXPECT_EQ(0, ids[1]);
  EXPECT_EQ(1, ids[0]);
  EXPECT_EQ(0u, reused_ids.size());

  HashMap::HashMapStruct items;
  EXPECT_EQ(0, hashmap.GetHashKeys(&items));

  EXPECT_EQ(2u, items.items.size());
  EXPECT_EQ(2u, items.counter);
  if (items.items[0].x == 1) {
    EXPECT_EQ(1, items.items[0].x);
    EXPECT_EQ(2, items.items[0].y);
    EXPECT_EQ(1, items.items[0].id);
    EXPECT_EQ(3, items.items[1].x);
    EXPECT_EQ(4, items.items[1].y);
    EXPECT_EQ(0, items.items[1].id);
  } else {
    EXPECT_EQ(1, items.items[1].x);
    EXPECT_EQ(2, items.items[1].y);
    EXPECT_EQ(1, items.items[1].id);
    EXPECT_EQ(3, items.items[0].x);
    EXPECT_EQ(4, items.items[0].y);
    EXPECT_EQ(0, items.items[0].id);
  }

  HashMap hashmap1(2);
  EXPECT_EQ(0, hashmap1.SetHashKeys(items));

  int64_t keys1[] = {1, 2, 3, 4, 13, 14};
  ret = hashmap1.Get(keys1, 3, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);   
  EXPECT_EQ(3u, ids.size());
  // alloc new id
  EXPECT_EQ(2, ids[2]);
  // old id
  EXPECT_EQ(0, ids[1]);
  EXPECT_EQ(1, ids[0]);
  EXPECT_EQ(0u, reused_ids.size());
}

TEST(HashMapTest, GetKeysAndSetKeysWithDel) {
  HashMap hashmap(1280);
  int64_t keys[] = {1, 2, 3, 4}; 
  vector<int64_t> ids;
  vector<int64_t> reused_ids;
  int ret = hashmap.Get(keys, 2, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);   
  EXPECT_EQ(2u, ids.size());
  // NOTE: the latter part of keys would get id first
  EXPECT_EQ(0, ids[1]);
  EXPECT_EQ(1, ids[0]);
  EXPECT_EQ(0u, reused_ids.size());
  
  int64_t del_keys1[] = {1, 2};
  EXPECT_EQ(0, hashmap.Del(del_keys1, 1, 2));

  HashMap::HashMapStruct items;
  EXPECT_EQ(0, hashmap.GetHashKeys(&items));

  EXPECT_EQ(1u, items.items.size());
  EXPECT_EQ(2u, items.counter);
  EXPECT_EQ(3, items.items[0].x);
  EXPECT_EQ(4, items.items[0].y);
  EXPECT_EQ(0, items.items[0].id);

  HashMap hashmap1(2);
  EXPECT_EQ(0, hashmap1.SetHashKeys(items));

  int64_t keys1[] = {1, 2, 3, 4, 13, 14};
  ret = hashmap1.Get(keys1, 3, 2, &ids, &reused_ids);
  ASSERT_EQ(0, ret);   
  EXPECT_EQ(3u, ids.size());
  EXPECT_EQ(1, ids[2]);
  EXPECT_EQ(0, ids[1]);
  EXPECT_EQ(2, ids[0]);
  EXPECT_EQ(1u, reused_ids.size());
  EXPECT_EQ(1, reused_ids[0]);
}
