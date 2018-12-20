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

#include "xdl/data_io/schema.h"
#include "gtest/gtest.h"

#include <stdlib.h>

namespace xdl {
namespace io {

TEST(DataIOTest, TestRead2ParseTxt) {
  Schema schema_;
  for (int c = 0; c < 5; ++c) {
    FeatureOption *s = new FeatureOption();
    s->set_name(std::to_string(c));
    s->set_type(c%2?kSparse:kDense);
    s->set_nvec(c);
    s->set_table(c%3);
    schema_.Add(s);
  }

  for (auto &kv: schema_.feature_opts()) {
    auto s = kv.second;
    EXPECT_STREQ(kv.first.c_str(), s->name().c_str());
    int c = atoi(s->name().c_str());
    EXPECT_EQ(c%2?kSparse:kDense, s->type());
    EXPECT_EQ(c, s->nvec());
    EXPECT_EQ(c%3, s->table());
  }

  const FeatureOption *fo = schema_.Get("3");
  EXPECT_NE(fo, nullptr);

  fo = schema_.Get("0", 0);
  EXPECT_NE(fo, nullptr);

  size_t st = schema_.ntable();
  EXPECT_EQ(st, 3);

  auto items = schema_.sparse_list();
  EXPECT_EQ(items.size(), 2);

  items = schema_.dense_list();
  EXPECT_EQ(items.size(), 3);
}

}
}
