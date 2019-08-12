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

#include <unistd.h>

#include "xdl/core/utils/logging.h"
#include "xdl/core/lib/timer.h"
#include "gtest/gtest.h"

namespace xdl {

TEST(TimerTest, TestTimer) {
  {
    XDL_TIMER_NOW(t1);
    XDL_TIMER_START(t1);
    XDL_TIMER_STOP(t1);
  }

  {
    tc_keeper::Instance()->ResetAll();
    tc_keeper::Instance()->Step(3);
    TimerCore *tc = tc_keeper::Instance()->Get("hello");
    XDL_CHECK(tc != nullptr);
    Timer *t1 = new Timer("hello", tc);
    XDL_CHECK(t1 != nullptr);
    t1->Start();
    sleep(1);
    t1->Reset();
    t1->Step(2);
    t1->Display();
    t1->Stop();
    delete t1;
  }

  {
    TimerCore *tc = tc_keeper::Instance()->Get("hello");
    XDL_CHECK(tc != nullptr);
    TimerScope *ts = new TimerScope("world", tc);
    XDL_CHECK(ts != nullptr);
    delete ts;
  }
}

}  // namespace xdl
