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

#include "xdl/data_io/scheduler.h"
#include "gtest/gtest.h"

#include <stdlib.h>

namespace xdl {
namespace io {

static const char *path = "sample.txt";
static const char *path2 = "sample2.txt";

TEST(DataIOTest, TestSchedule) {
  Scheduler sched(kLocal);
  sched.SetEpochs(2);

  sched.AddPath(path);
  sched.AddPath(path2);

  ASSERT_TRUE(sched.Schedule());

  ReadParam *rparam = nullptr;

  for (int i = 0; i < 6; ++i) {
    rparam = sched.Acquire();
    if (i >= 4) {
      ASSERT_EQ(nullptr, rparam) << "{" << i << "} " << rparam->DebugString();
      continue;
    }
    ASSERT_NE(nullptr, rparam);
    EXPECT_EQ(0, rparam->begin_);
    EXPECT_EQ(i<2?0:1, rparam->epoch_);
    EXPECT_GE(rparam->end_, 0);
    EXPECT_NE(nullptr, rparam->ant_);
    EXPECT_EQ(rparam->pathid_, i%2);
    EXPECT_STREQ(rparam->path_, i%2==0?path:path2);
    sched.Release(rparam);
  }
}


TEST(DataIOTest, TestRestore) {
  Scheduler sched(kLocal);
  sched.SetEpochs(2);

  sched.AddPath(path);
  sched.AddPath(path2);

  sched.Schedule();

  auto rparam = sched.Acquire();
  ASSERT_NE(nullptr, rparam);
  EXPECT_EQ(0, rparam->begin_);
  EXPECT_EQ(0, rparam->epoch_);
  EXPECT_GE(rparam->end_, 0);
  EXPECT_EQ(rparam->pathid_, 0);
  EXPECT_STREQ(rparam->path_, path);
  EXPECT_NE(nullptr, rparam->ant_);

  rparam->begin_ = 10;
  rparam->parsed_ = 1;

  sched.Release(rparam);

  rparam = sched.Acquire();
  ASSERT_NE(nullptr, rparam);
  EXPECT_EQ(0, rparam->begin_);
  EXPECT_EQ(0, rparam->epoch_);
  EXPECT_GE(rparam->end_, 0);
  EXPECT_EQ(rparam->pathid_, 1);
  EXPECT_STREQ(rparam->path_, path2);
  EXPECT_NE(nullptr, rparam->ant_);

  rparam->begin_ = 30;
  rparam->parsed_ = 3;

  DSState ds_state;
  sched.Store(&ds_state);

  EXPECT_EQ(2, ds_state.states_size());
  EXPECT_EQ(2, ds_state.epochs());
  for (int i = 0; i < ds_state.states_size(); ++i) {
    auto state = ds_state.states(i);
    EXPECT_EQ(i==0?3:0, state.begin());
    EXPECT_EQ(i==0?0:1, state.epoch());
    EXPECT_GE(state.end(), 0);
    EXPECT_FALSE(state.has_path());
    EXPECT_EQ(state.pathid(), i==0?1:0);
  }

  std::cout << ds_state.ShortDebugString() << std::endl;

  sched.Restore(ds_state);

  ASSERT_FALSE(sched.Schedule());

  for (int i = 0; i < 6; ++i) {
    rparam = sched.Acquire();
    if (i >= 3) {
      ASSERT_EQ(nullptr, rparam);
      continue;
    }
    ASSERT_NE(nullptr, rparam);
    if (i == 0) {
      EXPECT_EQ(3, rparam->begin_);
      EXPECT_EQ(3, rparam->parsed_);
      EXPECT_EQ(0, rparam->epoch_);
      EXPECT_EQ(rparam->pathid_, 1);
      EXPECT_STREQ(rparam->path_, path2);
    } else if (i == 1) {
      EXPECT_EQ(0, rparam->begin_);
      EXPECT_EQ(0, rparam->parsed_);
      EXPECT_EQ(1, rparam->epoch_);
      EXPECT_EQ(rparam->pathid_, 0);
      EXPECT_STREQ(rparam->path_, path);
    } else {
      EXPECT_EQ(0, rparam->begin_);
      EXPECT_EQ(0, rparam->parsed_);
      EXPECT_EQ(1, rparam->epoch_);
      EXPECT_EQ(rparam->pathid_, 1);
      EXPECT_STREQ(rparam->path_, path2);
    }
    EXPECT_GE(rparam->end_, 0);
    EXPECT_NE(nullptr, rparam->ant_);
    sched.Release(rparam);
  }
}

}
}
