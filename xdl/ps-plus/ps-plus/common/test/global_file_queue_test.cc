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
#include "ps-plus/common/global_file_queue.h"

TEST(GlobalFileQueueTest, TestNotInit) {
  ps::GlobalFileQueue queue;
  ps::WorkerState file;
  ps::Status st = queue.GetNextFile(0, &file);
  ASSERT_EQ(ps::Status::kFileQueueNeedWait, st.Code());
}

TEST(GlobalFileQueueTest, TestSimple) {
  ps::GlobalFileQueue queue;
  std::vector<std::string> paths = {"1.txt", "2.txt"};
  queue.Init(paths, 2, false);
  ps::WorkerState file;
  queue.GetNextFile(0, &file);
  ASSERT_EQ("1.txt", file.path_);
  ASSERT_EQ(0, file.begin_);
  ASSERT_EQ(0, file.end_);
  ASSERT_EQ(0, file.epoch_);
  queue.GetNextFile(0, &file);
  ASSERT_EQ("2.txt", file.path_);
  ASSERT_EQ(0, file.begin_);
  ASSERT_EQ(0, file.end_);
  ASSERT_EQ(0, file.epoch_);
  queue.GetNextFile(0, &file);
  ASSERT_EQ("1.txt", file.path_);
  ASSERT_EQ(0, file.begin_);
  ASSERT_EQ(0, file.end_);
  ASSERT_EQ(1, file.epoch_);
  queue.GetNextFile(0, &file);
  ASSERT_EQ("2.txt", file.path_);
  ASSERT_EQ(0, file.begin_);
  ASSERT_EQ(0, file.end_);
  ASSERT_EQ(1, file.epoch_);
  queue.GetNextFile(0, &file);
  ASSERT_EQ("", file.path_);
}

TEST(GlobalFileQueueTest, TestRestore) {
  {
    ps::GlobalFileQueue queue;
    std::vector<std::string> paths = {"1.txt", "2.txt", "3.txt", "4.txt"};
    ASSERT_TRUE(queue.Init(paths, 2, false).IsOk());
    ps::WorkerState file;
    queue.GetNextFile(0, &file);
    queue.GetNextFile(1, &file);
    queue.GetNextFile(0, &file);
    queue.GetNextFile(1, &file);
    std::vector<ps::WorkerState> worker_states;
    worker_states.push_back(ps::WorkerState(1, 11, 0, "1.txt"));
    worker_states.push_back(ps::WorkerState(3, 13, 0, "3.txt"));
    ASSERT_TRUE(queue.ReportWorkerState(0, worker_states).IsOk());
    worker_states.clear();
    worker_states.push_back(ps::WorkerState(2, 12, 0, "2.txt"));
    worker_states.push_back(ps::WorkerState(4, 14, 0, "4.txt"));
    ASSERT_TRUE(queue.ReportWorkerState(1, worker_states).IsOk());
    ASSERT_TRUE(queue.RestoreWorkerState(0).IsOk());
    ASSERT_TRUE(queue.RestoreWorkerState(1).IsOk());
    queue.GetNextFile(0, &file);
    ASSERT_EQ("1.txt", file.path_);
    ASSERT_EQ(1, file.begin_);
    ASSERT_EQ(11, file.end_);
    ASSERT_EQ(0, file.epoch_);
    queue.GetNextFile(0, &file);
    ASSERT_EQ("3.txt", file.path_);
    ASSERT_EQ(3, file.begin_);
    ASSERT_EQ(13, file.end_);
    ASSERT_EQ(0, file.epoch_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("2.txt", file.path_);
    ASSERT_EQ(2, file.begin_);
    ASSERT_EQ(12, file.end_);
    ASSERT_EQ(0, file.epoch_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("4.txt", file.path_);
    ASSERT_EQ(4, file.begin_);
    ASSERT_EQ(14, file.end_);
    ASSERT_EQ(0, file.epoch_);
    queue.GetNextFile(0, &file);
    ASSERT_EQ("1.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("2.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    queue.GetNextFile(0, &file);
    ASSERT_EQ("3.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("4.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    queue.GetNextFile(0, &file);
    ASSERT_EQ("", file.path_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("", file.path_);
  }

  {
    ps::GlobalFileQueue queue;
    std::vector<std::string> paths = {"1.txt", "2.txt", "3.txt", "4.txt"};
    ASSERT_TRUE(queue.Init(paths, 2, false).IsOk());
    ps::WorkerState file;
    queue.GetNextFile(0, &file);
    queue.GetNextFile(1, &file);
    queue.GetNextFile(0, &file);
    queue.GetNextFile(1, &file);
    std::vector<ps::WorkerState> worker_states;
    worker_states.push_back(ps::WorkerState(1, 11, 0, "1.txt"));
    ASSERT_TRUE(queue.ReportWorkerState(0, worker_states).IsOk());
    worker_states.clear();
    worker_states.push_back(ps::WorkerState(2, 12, 0, "2.txt"));
    ASSERT_TRUE(queue.ReportWorkerState(1, worker_states).IsOk());
    ASSERT_TRUE(queue.RestoreWorkerState(0).IsOk());
    ASSERT_TRUE(queue.RestoreWorkerState(1).IsOk());
    queue.GetNextFile(0, &file);
    ASSERT_EQ("1.txt", file.path_);
    ASSERT_EQ(1, file.begin_);
    ASSERT_EQ(11, file.end_);
    ASSERT_EQ(0, file.epoch_);
    queue.GetNextFile(0, &file);
    ASSERT_EQ("3.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(0, file.epoch_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("2.txt", file.path_);
    ASSERT_EQ(2, file.begin_);
    ASSERT_EQ(12, file.end_);
    ASSERT_EQ(0, file.epoch_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("4.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(0, file.epoch_);
    queue.GetNextFile(0, &file);
    ASSERT_EQ("1.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("2.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    queue.GetNextFile(0, &file);
    ASSERT_EQ("3.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("4.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    queue.GetNextFile(0, &file);
    ASSERT_EQ("", file.path_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("", file.path_);
  }
}

TEST(GlobalFileQueueTest, TestSerialize) {
  {
    ps::GlobalFileQueue queue;
    std::vector<std::string> paths = {"1.txt", "2.txt", "3.txt", "4.txt"};
    ASSERT_TRUE(queue.Init(paths, 2, false).IsOk());
    ps::WorkerState file;
    queue.GetNextFile(0, &file);
    queue.GetNextFile(1, &file);
    queue.GetNextFile(0, &file);
    queue.GetNextFile(1, &file);
    std::string buf;
    ASSERT_TRUE(queue.Serialize(&buf).IsOk());
    queue.GetNextFile(0, &file);
    ASSERT_EQ("1.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("2.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    ASSERT_TRUE(queue.Deserialize(buf).IsOk());
    queue.GetNextFile(0, &file);
    ASSERT_EQ("1.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("2.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
  }

  {
    ps::GlobalFileQueue queue;
    std::vector<std::string> paths = {"1.txt", "2.txt", "3.txt", "4.txt"};
    ASSERT_TRUE(queue.Init(paths, 2, false).IsOk());
    ps::WorkerState file;
    queue.GetNextFile(0, &file);
    queue.GetNextFile(1, &file);
    queue.GetNextFile(0, &file);
    queue.GetNextFile(1, &file);
    std::vector<ps::WorkerState> worker_states;
    worker_states.push_back(ps::WorkerState(1, 11, 0, "1.txt"));
    ASSERT_TRUE(queue.ReportWorkerState(0, worker_states).IsOk());
    worker_states.clear();
    worker_states.push_back(ps::WorkerState(2, 12, 0, "2.txt"));
    ASSERT_TRUE(queue.ReportWorkerState(1, worker_states).IsOk());
    std::string buf;
    ASSERT_TRUE(queue.Serialize(&buf).IsOk());

    worker_states.clear();
    worker_states.push_back(ps::WorkerState(3, 13, 0, "3.txt"));
    ASSERT_TRUE(queue.ReportWorkerState(0, worker_states).IsOk());
    worker_states.clear();
    worker_states.push_back(ps::WorkerState(4, 14, 0, "4.txt"));
    ASSERT_TRUE(queue.ReportWorkerState(1, worker_states).IsOk());

    ASSERT_TRUE(queue.Deserialize(buf).IsOk());
    ASSERT_TRUE(queue.RestoreWorkerState(0).IsOk());
    ASSERT_TRUE(queue.RestoreWorkerState(1).IsOk());

    queue.GetNextFile(0, &file);
    ASSERT_EQ("1.txt", file.path_);
    ASSERT_EQ(1, file.begin_);
    ASSERT_EQ(11, file.end_);
    ASSERT_EQ(0, file.epoch_);
    queue.GetNextFile(1, &file);
    ASSERT_EQ("2.txt", file.path_);
    ASSERT_EQ(2, file.begin_);
    ASSERT_EQ(12, file.end_);
    ASSERT_EQ(0, file.epoch_);
    queue.GetNextFile(0, &file);
    ASSERT_EQ("3.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(0, file.epoch_);
    queue.GetNextFile(1, &file);
    EXPECT_EQ("4.txt", file.path_);
    EXPECT_EQ(0, file.begin_);
    EXPECT_EQ(0, file.end_);
    EXPECT_EQ(0, file.epoch_);
    queue.GetNextFile(0, &file);
    ASSERT_EQ("1.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
    queue.GetNextFile(0, &file);
    ASSERT_EQ("2.txt", file.path_);
    ASSERT_EQ(0, file.begin_);
    ASSERT_EQ(0, file.end_);
    ASSERT_EQ(1, file.epoch_);
  }
}
