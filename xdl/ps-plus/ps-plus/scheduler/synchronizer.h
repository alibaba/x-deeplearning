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

#ifndef PS_SCHEDULER_SYNCHRONIZER_H_
#define PS_SCHEDULER_SYNCHRONIZER_H_

#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <memory>
#include <set>
#include <thread>

#include "ps-plus/common/status.h"

namespace ps {
namespace scheduler {

class SyncMechanism {
public:    
  SyncMechanism() {}
  virtual ~SyncMechanism() {}
  virtual void Reset() = 0;
  virtual Status WorkerReportFinish(int id) = 0;
};

class Asynchronizer : public SyncMechanism {
  struct Context {
    int id_;
    int step_;
    std::function<void (const Status&)> cb_;
  };
  int staleness_;
  int worker_count_;
  std::unique_ptr<Context[]> contexts_;
  std::map<int, std::set<Context*>> step_index_;
  std::mutex m_;
  std::set<int> removed_workers_;
  void UnlockNewSteps(int old_first);
public:
  Asynchronizer(int staleness, int worker_count);
  ~Asynchronizer();
  void Enter(int id, std::function<void (const Status&)> cb);
  Status WorkerReportFinish(int id);
  void Reset();  
};

class Synchronizer : public SyncMechanism {
  struct Context {
    int id_;
    std::function<void (int, const Status&)> cb_;
  };
  int worker_count_;
  int left_token_;
  int64_t current_token_;
  std::mutex m_;    
  std::unique_ptr<Context[]> contexts_;
  std::set<Context*> waiting_list_;
  std::set<int> working_list_;
  void UnlockNewToken();
public:    
  Synchronizer(int worker_count);
  ~Synchronizer() {}
  void Enter(int id, std::function<void (int64_t, const Status&)> cb);
  void Leave(int id, int64_t token, std::function<void (const Status&)> cb);
  Status WorkerReportFinish(int id);
  void Reset();
};

} // namespace scheduler
} // namespace ps

#endif // PS_SCHEDULER_SYNCHRONIZER_H_
