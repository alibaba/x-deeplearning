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

// Copyright 2018 Alibaba Inc. All Rights Reserved

#include "tdm/tair_store.h"

#include "gtest/gtest.h"

namespace tdm {

TEST(TairStore, TestInit) {
  TairStore store;
  ASSERT_FALSE(store.Init("aaaa"));
  ASSERT_TRUE(store.Init("master=10.97.212.23:5198;slave=10.97.212.23:5198;"
                        "group=algo_test;area=598"));
}

TEST(TairStore, TestGet) {
  TairStore store;
  ASSERT_TRUE(store.Init("master=10.97.212.23:5198;slave=10.97.212.23:5198;"
                         "group=algo_test;area=598"));
  store.Remove("abc");
  std::string value;
  ASSERT_FALSE(store.Get("abc", &value));
  ASSERT_TRUE(store.Put("abc", "abcd"));
  ASSERT_TRUE(store.Get("abc", &value));
  ASSERT_EQ("abcd", value);
  ASSERT_TRUE(store.Put("add1", "add1"));
  ASSERT_TRUE(store.Put("add2", "add2"));
}

TEST(TairStore, TestPut) {
  TairStore store;
  ASSERT_TRUE(store.Init("master=10.97.212.23:5198;slave=10.97.212.23:5198;"
                         "group=algo_test;area=598"));
  store.Remove("abc");
  ASSERT_TRUE(store.Put("abc", "abcdefghijkl"));
  std::string value;
  store.Get("abc", &value);
  ASSERT_EQ("abcdefghijkl", value);
  ASSERT_TRUE(store.Put("changshengshengwu", "yimiaozaojia, tianliburong"));
  store.Get("changshengshengwu", &value);
  ASSERT_EQ("yimiaozaojia, tianliburong", value);
}

TEST(TairStore, TestMPut) {
  TairStore store;
  ASSERT_TRUE(store.Init("master=10.97.212.23:5198;slave=10.97.212.23:5198;"
                         "group=algo_test;area=598"));
  std::vector<std::string> keys;
  for (int i = 0; i < 1024; ++i) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "test%d", i);
    keys.push_back(buffer);
  }

  auto result = store.MPut(keys, keys);
  for (auto it = result.begin(); it != result.end(); ++it) {
    ASSERT_TRUE(*it);
  }
}


TEST(TairStore, TestMGet) {
  TairStore store;
  ASSERT_TRUE(store.Init("master=10.97.212.23:5198;slave=10.97.212.23:5198;"
                         "group=algo_test;area=598"));
  std::vector<std::string> keys;
  for (int i = 0; i < 300; ++i) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "test%d", i);
    keys.push_back(buffer);
  }

  std::vector<std::string> values;
  auto result = store.MGet(keys, &values);
  for (size_t i = 0; i < keys.size(); ++i) {
    ASSERT_TRUE(result[i]);
    ASSERT_EQ(keys[i], values[i]);
  }
}

}  // namespace tdm
