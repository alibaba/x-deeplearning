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
#include <algorithm>
#include <random>

#include "xdl/core/lib/blocking_queue.h"
#include "gtest/gtest.h"

namespace xdl {

struct Item {
  int a;
  void* b;
};

TEST(TestBlockQueue, All) {
  BlockingQueue<Item> bq(2, 1.0);
  EXPECT_TRUE(bq.Empty());

  Item item = { 1, (void*)0x0002 };
  bq.Enqueue(item);
  EXPECT_EQ(bq.Size(), 1);

  Item item2 = { 2, (void*)0x0003 };
  bq.Enqueue(item2);
  EXPECT_EQ(bq.Size(), 2);

  EXPECT_TRUE(bq.Full());
  EXPECT_FALSE(bq.Empty());

  EXPECT_TRUE(bq.TryDequeue(&item, 0));
  bq.EnqueueFront(item);

  EXPECT_TRUE(bq.TryDequeue(&item, 0));
  EXPECT_EQ(item.a, 1);
  EXPECT_EQ(item.b, (void*)0x0002);
}

bool KLess(const Item &n1, const Item &n2) {
  return n1.a > n2.a;
}

TEST(TestBlockPriorityQueue, All) {
  const size_t N = 4;
  BlockingQueue<Item> pq(N, 1.0, KLess);
  EXPECT_TRUE(pq.Empty());

  std::vector<int> vec;
  for (size_t i = 0; i < N; ++i) {
    vec.push_back(i);
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(vec.begin(), vec.end(), g);

  for (size_t i = 0; i < N; ++i) {
    int a = vec[i];
    Item item;
    item.a = a;
    item.b = (void*)(size_t)a;
    pq.Enqueue(item);
  }

  EXPECT_TRUE(pq.Full());

  for (size_t i = 0; i < N; ++i) {
    Item item = pq.Dequeue();
    EXPECT_EQ(i, item.a);
    EXPECT_EQ((void*)i, item.b);
  }

  EXPECT_TRUE(pq.Empty());
}

TEST(TestMultiBlockingQueue, All) {
  MultiBlockingQueue<Item> mbq(2, 1.0);
  Item item = { 1, (void*)0x0001 };
  mbq.Enqueue("A", item);
  mbq.Enqueue("B", item);
  
  item = mbq.Dequeue("A");
  EXPECT_EQ(item.a, 1);
  EXPECT_EQ(item.b, (void*)0x0001);

  EXPECT_EQ(mbq.QueueNum(), 2);
  EXPECT_FALSE(mbq.Full("A"));
  EXPECT_FALSE(mbq.Full("B"));

  std::set<std::string> names = mbq.Names();
  EXPECT_EQ(names.size(), 2);
  EXPECT_TRUE(names.count("A") == 1);
  EXPECT_TRUE(names.count("B") == 1);
}

}  // namespace xdl
