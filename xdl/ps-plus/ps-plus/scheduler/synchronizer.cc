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

#include "synchronizer.h"

#include "ps-plus/common/status.h"
#include <glog/logging.h>
#include <string>
#include <iostream>

using namespace std;
using namespace std::chrono;
using namespace ps;
using namespace ps::scheduler;

namespace ps {
namespace scheduler {

namespace {

function<void (int, const Status&)> MkCb(int id) {
  return [id](int, const Status& st) {
    LOG(WARNING) << "Null callback for worker " << id << " invoked";
  };
}

}

Synchronizer::Synchronizer(int worker_count) : worker_count_(worker_count), left_token_(worker_count), current_token_(0) {
  contexts_.reset(new Context[worker_count_ + 100]);
  Context* ctxs = contexts_.get();
  for (int i = 0; i < worker_count_ + 100; i++) {
    Context* p = ctxs + i;
    p->id_ = i;
    p->cb_ = MkCb(i);
  }
}

void Synchronizer::UnlockNewToken() {
  left_token_ = worker_count_;
  current_token_++;
  while(left_token_ > 0 && !waiting_list_.empty()) {
      auto iter = waiting_list_.begin();
      (*iter)->cb_(current_token_, Status::Ok());
      working_list_.insert((*iter)->id_);
      left_token_--;
      waiting_list_.erase(iter);
  }
}

void Synchronizer::Enter(int id, function<void (int64_t, const Status&)> cb) {
  if (working_list_.find(id) != working_list_.end()) {
    cb(current_token_, Status::Ok());
    return;
  }
  if (left_token_ > 0) {
    left_token_--;
    working_list_.insert(id);
    cb(current_token_, Status::Ok());
    return;
  }
  Context* ctx = contexts_.get() + id;
  ctx->cb_ = cb;
  waiting_list_.insert(ctx);
}

Status Synchronizer::WorkerReportFinish(int id) {
  if (working_list_.find(id) == working_list_.end()) {
    return Status::Ok();
  }
  working_list_.erase(id);
  if (left_token_ == 0 && working_list_.empty()) {
    UnlockNewToken();
  }
  return Status::Ok();
}

void Synchronizer::Leave(int id, int64_t token, function<void (const Status&)> cb) {
  if (token != current_token_) {
    LOG(WARNING) << "Receive token " << token << " from " << id << " while current_token_ is " << current_token_;
    cb(Status::Ok());
  }
  if (working_list_.find(id) == working_list_.end()) {
    LOG(FATAL) << "Worker " << id << " not granted token, but it call leave with token " << token << ", current token is " << current_token_;
    abort();
  }
  working_list_.erase(id);
  if (left_token_ == 0 && working_list_.empty()) {
    UnlockNewToken();
  }
  cb(Status::Ok());
}

void Synchronizer::Reset() {
  for (auto iter : waiting_list_) {
    iter->cb_(-1, Status::Ok());
  }
  waiting_list_.clear();
  working_list_.clear();
  current_token_++;
  left_token_ = worker_count_;
}

} // namespace scheduler
} // namespace ps
