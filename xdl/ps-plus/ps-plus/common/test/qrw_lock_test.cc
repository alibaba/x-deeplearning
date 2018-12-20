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
#include "ps-plus/common/qrw_lock.h"

#include <algorithm>
#include <thread>
#include <vector>

using ps::QRWLock;
using ps::QRWLocker;

TEST(QRWLock, QRWLock) {
  int x[128];
  for (int i = 0; i < 128; i++) {
    x[i] = 0;
  }
  QRWLock lock;
  bool success = true;
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; i++) {
    threads.emplace_back([&x, &lock, &success](){
      int a[128];
      for (int i = 0; i < 128; i++) {
        a[i] = i;
      }
      for (int i = 0; i < 128; i++) {
        std::swap(a[128 - i - 1], a[rand() % (128 - i)]);
      }
      for (int i = 0; i < 1000; i++) {
        QRWLocker locker(lock, QRWLocker::kSimpleRead);
        for (int j = 0; j < 127; j++) {
          if (x[a[j]] != x[a[j + 1]]) {
            success = false;
          }
        }
      }
    });
  }
  for (int i = 0; i < 10; i++) {
    threads.emplace_back([&x, &lock](){
      int a[128];
      for (int i = 0; i < 128; i++) {
        a[i] = i;
      }
      for (int i = 0; i < 128; i++) {
        std::swap(a[128 - i - 1], a[rand() % (128 - i)]);
      }
      for (int i = 0; i < 1000; i++) {
        QRWLocker locker(lock, QRWLocker::kWrite);
        int b = rand();
        for (int j = 0; j < 128; j++) {
          x[a[j]] = b;
        }
      }
    });
  }
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
  EXPECT_TRUE(success);
}

