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
#include "ps-plus/scheduler/synchronizer.h"
#include <unistd.h>
#include <iostream>
#include <mutex>

using namespace std;
using namespace std::chrono;
using namespace ps;
using namespace ps::scheduler;

TEST(Synchronizer, EnterAndLeave) {
  unique_ptr<Synchronizer> sync(new Synchronizer(3));
  int64_t result = -1;
  sync->Enter(0, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 0);
  sync->Enter(0, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 0);
  sync->Enter(1, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 0);
  sync->Enter(2, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 0);

  string execute_log;
  sync->Leave(2, 0, [&execute_log](const Status& st) {
    execute_log += "2L";
  });
  sync->Enter(2, [&execute_log, &result](int token, const Status& st) {
    execute_log += "2E";
    result = token;
  });
  EXPECT_EQ(result, 0);    
  sync->Leave(1, 0, [&execute_log](const Status& st) {
    execute_log += "1L";
  });
  sync->Leave(0, 0, [&execute_log](const Status& st) {
    execute_log += "0L";                            
  });
  EXPECT_EQ(execute_log, "2L1L2E0L");
  EXPECT_EQ(result, 1);      
}

TEST(Synchronizer, Reset) {
  unique_ptr<Synchronizer> sync(new Synchronizer(3));
  int64_t result = -1;
  sync->Enter(0, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 0);
  sync->Enter(0, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 0);
  sync->Enter(1, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 0);

  sync->Reset();
  
  sync->Enter(2, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 1);
  sync->Enter(0, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 1);
  sync->Enter(1, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 1);
  sync->Enter(1, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 1);      
}

TEST(Synchronizer, WorkerReportFinish) {
  unique_ptr<Synchronizer> sync(new Synchronizer(2));
  int64_t result = -1;
  sync->Enter(0, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 0);
  sync->Enter(1, [&result](int64_t token, const Status& st) {
    result = token;
  });
  EXPECT_EQ(result, 0);
  string execute_log;
  sync->Leave(0, 0, [&execute_log](const Status& st) {
    execute_log += "0L";
  });
  EXPECT_EQ(execute_log, "0L");
  sync->Enter(0, [&result, &execute_log](int64_t token, const Status& st) {
    result = token;    
    execute_log += "0E";
  });
  EXPECT_EQ(result, 0L);
  EXPECT_EQ(execute_log, "0L");
  sync->WorkerReportFinish(1);
  EXPECT_EQ(execute_log, "0L0E");
  EXPECT_EQ(result, 1L);    
}
