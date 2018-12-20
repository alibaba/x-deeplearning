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
#include "ps-plus/common/memblock.h"

using ps::SMemBlock;
using ps::MemBlockPool;

TEST(SMemBlockTest, Get) {
  SMemBlock<3> block;
  ASSERT_EQ(0, block.Push(1, 1, 2));
  ASSERT_EQ(0, block.Push(2, 2, 3));
  ASSERT_EQ(0, block.Push(3, 3, 4));
  auto item = block.Get(1, 1);
  EXPECT_TRUE(nullptr != item);
  EXPECT_EQ(1, item->x);
  EXPECT_EQ(1, item->y);
  EXPECT_EQ(2, item->id);
  item = block.Get(2, 2);
  EXPECT_TRUE(nullptr != item);
  EXPECT_EQ(2, item->x);
  EXPECT_EQ(2, item->y);
  EXPECT_EQ(3, item->id);
  item = block.Get(3, 3);
  EXPECT_TRUE(nullptr != item);
  EXPECT_EQ(3, item->x);
  EXPECT_EQ(3, item->y);
  EXPECT_EQ(4, item->id);
  item = block.Get(4, 4);
  EXPECT_TRUE(nullptr == item);
}

TEST(SMemBlockTest, Del) {
  SMemBlock<5> block;
  ASSERT_EQ(0, block.Push(1, 1, 2));
  ASSERT_EQ(0, block.Push(2, 2, 3));
  ASSERT_EQ(0, block.Push(3, 3, 4));
  int64_t val;
  EXPECT_EQ(-1, block.Del(4, 4, &val));  
  // the order of key list: 3->2->1
  // del the head
  EXPECT_EQ(0, block.Del(3, 3, &val));
  EXPECT_EQ(4, val);
  auto item = block.Get(2, 2);
  ASSERT_TRUE(nullptr != item);
  EXPECT_EQ(2, item->x); 
  EXPECT_EQ(2, item->y); 
  EXPECT_EQ(3, item->id);
  // del the tail
  EXPECT_EQ(0, block.Del(1, 1, &val));
  EXPECT_EQ(2, val);
  item = block.Get(2, 2);
  ASSERT_TRUE(nullptr != item);
  EXPECT_EQ(2, item->x); 
  EXPECT_EQ(2, item->y); 
  EXPECT_EQ(3, item->id);
  // del the left one
  EXPECT_EQ(0, block.Del(2, 2, &val));
  EXPECT_EQ(3, val);
}

TEST(SMemBlockTest, Push) {
  SMemBlock<2> block;
  EXPECT_EQ(0, block.Push(1, 1, 2));
  EXPECT_EQ(0, block.Push(2, 2, 3));
  EXPECT_NE(0, block.Push(3, 3, 4));
}

TEST(MemBlockPoolTest, Borrow) {
  MemBlockPool<10> pool(2);
  auto block = pool.Borrow();
  EXPECT_TRUE(nullptr != block);
  EXPECT_TRUE(nullptr == block->Next());
  block = pool.Borrow();
  EXPECT_TRUE(nullptr != block);
  EXPECT_TRUE(nullptr == block->Next());
  block = pool.Borrow();
  EXPECT_TRUE(nullptr == block);
}

TEST(MemBlockPoolTest, Return) {
  MemBlockPool<10> pool(2);
  auto block1 = pool.Borrow();
  EXPECT_TRUE(nullptr != block1);
  auto block2 = pool.Borrow();
  EXPECT_TRUE(nullptr != block2);
  pool.Return(block2); 
  auto block3 = pool.Borrow();
  EXPECT_TRUE(nullptr != block3);
  auto block4 = pool.Borrow();
  EXPECT_TRUE(nullptr == block4);
}
