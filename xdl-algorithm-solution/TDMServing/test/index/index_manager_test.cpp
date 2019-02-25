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

#define protected public
#define private public

#include "index/index_manager.h"
#include "index/index.h"
#include "proto/search.pb.h"

namespace tdm_serving {

TEST(IndexManager, init_and_search) {
  std::string conf_path = "test_data/conf/index_manager.conf";

  // init
  ASSERT_TRUE(IndexManager::Instance().Init(conf_path));

  Index* index = IndexManager::Instance().GetIndex("mock_index_disable");
  EXPECT_STREQ(nullptr, reinterpret_cast<const char*>(index));

  index = IndexManager::Instance().GetIndex("mock_index_no_version");
  EXPECT_EQ("mock_index_no_version", index->index_name());

  index = IndexManager::Instance().GetIndex("mock_index");
  EXPECT_EQ("mock_index", index->index_name());

  // search
  SearchParam search_param;
  search_param.set_index_name("mock_index");

  SearchResult search_result;

  ASSERT_TRUE(IndexManager::Instance().Search(search_param, &search_result));
}

}  // namespace tdm_serving
