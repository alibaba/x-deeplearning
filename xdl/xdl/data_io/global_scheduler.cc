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

#include "xdl/data_io/global_scheduler.h"

#include <string.h>
#include <chrono>
#include <thread>

#include "ps-plus/client/base_client.h"
#include "xdl/core/ops/ps_ops/client.h"

namespace xdl {
namespace io {

GlobalScheduler::GlobalScheduler(
    FileSystem *fs, 
    const std::string& name,
    size_t epochs, 
    size_t worker_id) 
  : Scheduler(fs, epochs)
  , name_(name)
  , client_(nullptr)
  , epoch_isolate_(false)
  , worker_id_(worker_id) {
  XDL_CHECK(fs != nullptr);
  fs_ = fs;
  XDL_CHECK(GetClient(&client_).IsOk());
}

GlobalScheduler::GlobalScheduler(
    const std::string& name,
    FSType fs_type, 
    const std::string &namenode, 
    size_t epochs,
    size_t worker_id)
  : Scheduler(fs_type, namenode, epochs)
  , name_(name)
  , client_(nullptr)
  , epoch_isolate_(false)
  , worker_id_(worker_id) {
  fs_ = GetFileSystem(
      fs_type, 
      namenode.empty() ? nullptr : namenode.c_str());
  XDL_CHECK(GetClient(&client_).IsOk());
}

bool GlobalScheduler::SetEpochIsolate(bool value) {
  epoch_isolate_ = value;
  return true;
}

bool GlobalScheduler::Schedule() {
  finished_ = false;
  XDL_CHECK(client_->InitGlobalQueue(
            name_, paths_, epochs_, 
            epoch_isolate_).IsOk());
  return true;
}

ReadParam *GlobalScheduler::Acquire() {
  if (finished_) {
    return nullptr;
  }

  std::string path;
  size_t begin;
  size_t epoch;
  ps::Status st = client_->GetNextFile(
      name_, worker_id_, &path, &begin, &epoch);
  while (st.Code() == ps::Status::kFileQueueNeedWait) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    st = client_->GetNextFile(name_, worker_id_, &path, &begin, &epoch);    
  }
  
  XDL_CHECK(st.IsOk());
  if (path.empty()) {
    finished_ = true;
    return nullptr;
  }

  ReadParam *rparam = new ReadParam();
  rparam->path_ = strdup(path.c_str());
  rparam->epoch_ = epoch;
  rparam->begin_ = begin;
  rparam->ant_ = fs_->GetZAnt(rparam->path_, ztype_);
  if (ztype_ == kZLib) {
    rparam->end_ = ULONG_MAX;
  } else {
    rparam->end_ = fs_->Size(rparam->path_);
  }

  XDL_CHECK(rparam->end_ > rparam->begin_);
  std::unique_lock<std::mutex> lck(mutex_);
  using_.insert(rparam);
  XDL_LOG(DEBUG) << "acquire " << rparam->DebugString();
  return rparam;
}

void GlobalScheduler::Release(ReadParam *rparam) {
  XDL_LOG(DEBUG) << "release " << rparam->DebugString();
  XDL_CHECK(rparam != nullptr);
  std::unique_lock<std::mutex> lck(mutex_);
  auto it = using_.find(rparam);
  if (it != using_.end()) {
    using_.erase(it);
  }

  delete rparam;
}

bool GlobalScheduler::Store(DSState *ds_state) {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<ps::WorkerState> worker_states;
  for (auto &it : using_) {
    ps::WorkerState ws;
    ws.begin_ = it->begin_;
    ws.end_ = it->end_;
    ws.epoch_ = it->epoch_;
    ws.path_ = it->path_;    
    worker_states.push_back(ws);
  }

  XDL_CHECK(client_->ReportWorkerState(
            name_, worker_id_, worker_states).IsOk());
  return true;
}

bool GlobalScheduler::Restore(const DSState &ds_state) {
  std::unique_lock<std::mutex> lock(mutex_);
  XDL_CHECK(client_->RestoreWorkerState(name_, worker_id_).IsOk());
  // for (auto& it: using_) {
  //   delete it;
  // }

  using_.clear();
  XDL_LOG(DEBUG) << "global scheduler restore worker id:" << worker_id_;
  return true;
}

}  // namespace io
}  // namespace xdl
