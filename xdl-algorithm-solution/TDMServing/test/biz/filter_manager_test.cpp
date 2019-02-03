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

#include "biz/filter_manager.h"
#include "biz/filter.h"
#include "proto/search.pb.h"
#include "util/str_util.h"
#include "util/log.h"

namespace tdm_serving {

class MockFilter : public Filter {
 public:
  virtual bool IsFiltered(const FilterInfo* /*filter_info*/,
                          const Item& /*item*/) {
    LOG_INFO << "Mock Filter";
    return false;
  }
};

REGISTER_FILTER(mock_filter, MockFilter);

TEST(FilterManager, init_and_get_filter) {
  std::string conf_path = "test_data/conf/filter_manager.conf";

  // init
  ASSERT_TRUE(FilterManager::Instance().Init(conf_path));

  Filter* filter = FilterManager::Instance().GetFilter("mock_filter");
  EXPECT_EQ("mock_filter", filter->filter_name());
}

}  // namespace tdm_serving
