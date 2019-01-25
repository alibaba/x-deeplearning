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

TEST(Asynchronizer, EnterAndFinish) {
  unique_ptr<Asynchronizer> async(new Asynchronizer(1, 3));
  string execute_log;
  async->Enter(0, [&execute_log](const Status& st) {
    execute_log += "0-0";
  });
  async->Enter(0, [&execute_log](const Status& st) {
    execute_log += "0-1";
  });
  async->Enter(1, [&execute_log](const Status& st) {
    execute_log += "1-0";
  });
  async->Enter(2, [&execute_log](const Status& st) {
    execute_log += "2-0";
  });    
  EXPECT_EQ(execute_log, "0-01-00-12-0");
  async->Enter(0, [&execute_log](const Status& st) {
    execute_log += "0-2";
  });
  EXPECT_EQ(execute_log, "0-01-00-12-0");
  Status st = async->WorkerReportFinish(1);
  EXPECT_TRUE(st.IsOk());  
  EXPECT_EQ(execute_log, "0-01-00-12-0");
  st = async->WorkerReportFinish(2);
  EXPECT_TRUE(st.IsOk());
  EXPECT_EQ(execute_log, "0-01-00-12-00-2");
}

TEST(Asynchronizer, Reset) {
  unique_ptr<Asynchronizer> async(new Asynchronizer(1, 3));
  
  string execute_log;
  async->Enter(0, [&execute_log](const Status& st) {
    execute_log += "0-0";
  });
  async->Enter(0, [&execute_log](const Status& st) {
    execute_log += "0-1";
  });
  async->Enter(1, [&execute_log](const Status& st) {
    execute_log += "1-0";
  });
  async->Enter(2, [&execute_log](const Status& st) {
    execute_log += "2-0";
  });    
  EXPECT_EQ(execute_log, "0-01-00-12-0");

  async->Reset();
  execute_log = "";

  async->Enter(0, [&execute_log](const Status& st) {
    execute_log += "0-0";
  });
  async->Enter(0, [&execute_log](const Status& st) {
    execute_log += "0-1";
  });
  async->Enter(1, [&execute_log](const Status& st) {
    execute_log += "1-0";
  });
  async->Enter(2, [&execute_log](const Status& st) {
    execute_log += "2-0";
  });    
  EXPECT_EQ(execute_log, "0-01-00-12-0");  
}

