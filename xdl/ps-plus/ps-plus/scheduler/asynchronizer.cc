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

#include <string>
#include <iostream>

using namespace std;
using namespace std::chrono;
using namespace ps;
using namespace ps::scheduler;

namespace ps {
namespace scheduler {

namespace {

function<void (const Status&)> MkCb(int id) {
  return [id](const Status& st) {
  };
}

}

Asynchronizer::Asynchronizer(int staleness, int worker_count) : staleness_(staleness), worker_count_(worker_count) {
  contexts_.reset(new Context[worker_count_]);
  Context* ctxs = contexts_.get();
  for (int i = 0; i < worker_count_; i++) {
    Context* p = ctxs + i;
    p->id_ = i;
    p->step_ = 0;
    p->cb_ = MkCb(i);
    step_index_[0].insert(p);
  }
}

Asynchronizer::~Asynchronizer() {
}

void Asynchronizer::UnlockNewSteps(int least_step) {
  int step = least_step + staleness_ + 1;
  auto it = step_index_.find(step);
  if (it == step_index_.end()) { return; }
  for (Context* p: it->second) {
    p->cb_(Status::Ok());
    p->cb_ = MkCb(p->id_);
  }
}

void Asynchronizer::Enter(int id, function<void (const Status&)> cb) {
  if (id < 0 || id >= worker_count_) {
    cb(Status::ArgumentError("Offset out of bound: min=0, max="
                             + to_string(worker_count_) + ", actual="
                             + to_string(id)));
    return;
  }
  if (removed_workers_.find(id) != removed_workers_.end()) {
    abort();
  }
  Context* ctx = contexts_.get() + id;
  step_index_[ctx->step_].erase(ctx);
  if (step_index_[ctx->step_].size() == 0) {
    int least_step = step_index_.begin()->first;
    step_index_.erase(ctx->step_);
    if (ctx->step_ == least_step) {
      UnlockNewSteps(least_step);
    }
  }
  ctx->step_++;
  step_index_[ctx->step_].insert(ctx);
  if (ctx->step_ > step_index_.begin()->first + staleness_) {
    ctx->cb_ = cb;
    return;
  }
  cb(Status::Ok());
}

Status Asynchronizer::WorkerReportFinish(int id) {
  if (id < 0 || id >= worker_count_) {
    return Status::ArgumentError("Offset out of bound: min=0, max="
                                 + to_string(worker_count_) + ", actual="
                                 + to_string(id));
  }
  removed_workers_.insert(id);
  Context* ctx = contexts_.get() + id;  
  step_index_[ctx->step_].erase(ctx);
  if (step_index_[ctx->step_].size() == 0) {
    int least_step = step_index_.begin()->first;
    step_index_.erase(ctx->step_);
    if (ctx->step_ == least_step) {
      UnlockNewSteps(least_step);
    }
  }  
  return Status::Ok();
}

void Asynchronizer::Reset() {
  for (int i = 0; i < worker_count_; i++) {
    Context* p = contexts_.get() + i;
    p->cb_(Status::Ok());
  }
  step_index_.clear();
  contexts_.reset(new Context[worker_count_]);
  Context* ctxs = contexts_.get();
  for (int i = 0; i < worker_count_; i++) {
    Context* p = ctxs + i;
    p->id_ = i;
    p->step_ = 0;
    p->cb_ = MkCb(i);
    if (removed_workers_.find(i) == removed_workers_.end()) {
      step_index_[0].insert(p);
    }
  }
}

} // namespace scheduler
} // namespace ps
