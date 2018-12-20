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

#include "xdl/data_io/sgroup.h"
#include "gtest/gtest.h"

namespace xdl {
namespace io {

TEST(SGroupTest, TestSGroup) {
  SGroup sgroup;
  auto sg_ = sgroup.New();
  EXPECT_NE(nullptr, sg_);
  EXPECT_TRUE(sgroup.own_);

  auto sg = sgroup.Get();
  EXPECT_EQ(sg_, sg);


  auto ft = sg_->add_feature_tables();

  int c = 5;
  for (int i = 0; i < c; ++i) {
    auto fl = ft->add_feature_lines();
    auto label = sg_->add_labels();
  }

  /// reset
  sgroup.Reset();
  EXPECT_EQ(0, sgroup.begin_);
  EXPECT_EQ(c, sgroup.end_);
  EXPECT_EQ(c, sgroup.size_);

  /// reset begin end
  int b = 1;
  int e = c-2;
  sgroup.Reset(b, e);
  EXPECT_EQ(b, sgroup.begin_);
  EXPECT_EQ(e, sgroup.end_);
  EXPECT_EQ(c, sgroup.size_);

  /// clone tail
  SGroup tail;
  tail.CloneTail(&sgroup);
  EXPECT_EQ(e, tail.begin_);
  EXPECT_EQ(c, tail.end_);
  EXPECT_EQ(c, tail.size_);

  SGroup tail1;
  tail.CloneTail(&sgroup, c-1);
  EXPECT_EQ(e, tail.begin_);
  EXPECT_EQ(c-1, tail.end_);
  EXPECT_EQ(c, tail.size_);

  /// copy constructor
  SGroup me;
  me.New();
  auto sample = me.Get();
  EXPECT_NE(sample, nullptr);
}

}
}
