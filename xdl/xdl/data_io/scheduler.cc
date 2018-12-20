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

#include <string.h>

#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

Scheduler::Scheduler(FileSystem *fs, size_t epochs) 
    : epochs_(epochs), finished_(true), rparams_(kSchedCap) {
  XDL_CHECK(fs != nullptr);
  fs_ = fs;
}

Scheduler::Scheduler(FSType fs_type, const std::string &namenode, size_t epochs)
    : epochs_(epochs), finished_(true), rparams_(kSchedCap) {
  fs_ = GetFileSystem(fs_type, namenode.empty()?nullptr:namenode.c_str());
}

bool Scheduler::AddPath(const std::string &path) {
  paths_.push_back(path);
  /*
  if (fs_->IsDir(path.c_str())) {
    const auto &paths = fs_->Dir(path.c_str());
    for (auto &path: paths) {
      paths_.insert(path);
    }
  } else if (fs_->IsReg(path.c_str())) {
    paths_.insert(path);
  } else {
    XDL_LOG(ERROR) << path << " not exist";
    return false;
  }
  */
  return true;
}

bool Scheduler::SetEpochs(size_t epochs) {
  XDL_CHECK(epochs < kEpochMax);
  epochs_ = epochs;
  return true;
}

std::vector<ReadParam *>Scheduler::Schedule(const char *path, size_t epochs) {
  std::vector<ReadParam *> rparams;
  size_t size = fs_->Size(path);
  for (size_t epoch = 0; epoch < (epochs==0?1:epochs); ++epoch) {
    ReadParam *rparam = new ReadParam();
    rparam->path_ = strdup(path);
    rparam->epoch_ = epoch;
    rparam->ant_ = (epoch == 0 ? fs_->GetAnt(path) : nullptr);  /// open later
    rparam->begin_ = 0;
    rparam->end_ = size;
    XDL_CHECK(rparam->end_ > rparam->begin_);

    rparams.push_back(rparam);
  }

  return rparams;
}

bool Scheduler::Schedule() {
  finished_ = false;
  std::unique_lock<std::mutex> lck(mutex_);
  if (restored_) {
    XDL_LOG(DEBUG) << "Has been restored, no need to scheduled again";
    return false;
  }
  if (rparams_.Size() > 0) {
    XDL_LOG(DEBUG) << "Already scheduled, size=" << rparams_.Size();
    return false;
  }
  std::vector<std::vector<ReadParam *>>rparams_list;
  for (auto &path: paths_) {
    std::vector<ReadParam *>rparams = Schedule(path.c_str(), epochs_);
    assert(rparams.size() == epochs_==0?1:epochs_);
    rparams_list.push_back(rparams);
  }

  size_t count = 0;
  for (size_t epoch = 0; epoch < (epochs_==0?1:epochs_); ++epoch) {
    for (size_t i = 0; i < rparams_list.size(); ++i, ++count) {
      ReadParam *rparam = rparams_list[i][epoch];
      rparams_.Enqueue(rparam);
      XDL_LOG(DEBUG) << "schedule " << rparam->DebugString();
    }
  }
  if (epochs_ != 0 || count == 0) {
    /// end of data
    rparams_.Enqueue(nullptr);
    XDL_LOG(DEBUG) << "schedule nullptr as end";
  }
  return true;
}

ReadParam *Scheduler::Acquire() {
  if (finished_) {
    return nullptr;
  }
  ReadParam *rparam = rparams_.Dequeue();
  if (rparam == nullptr) {
    finished_ = true;
    return nullptr;
  }
  if (rparam->ant_ == nullptr) {
    rparam->ant_ = fs_->GetAnt(rparam->path_);
  }
  std::unique_lock<std::mutex> lck(mutex_);
  using_.insert(rparam);
  XDL_LOG(DEBUG) << "acquire " << rparam->DebugString();
  return rparam;
}

void Scheduler::Release(ReadParam *rparam) {
  XDL_LOG(DEBUG) << "release " << rparam->DebugString();
  XDL_CHECK(rparam != nullptr);

  /* erase from using */
  std::unique_lock<std::mutex> lck(mutex_);
  auto it = using_.find(rparam);
  XDL_CHECK(it != using_.end());
  using_.erase(it);

  /* reuse or delete */
  if (epochs_ == 0) {
    rparam->begin_ = 0;
    XDL_CHECK(rparam->ant_ != nullptr);
    rparam->ant_->Seek(0);
    rparams_.Enqueue(rparam);
    XDL_LOG(DEBUG) << "re schedule " << rparam->DebugString();
  } else {
    delete rparam;
  }
}

bool Scheduler::finished() const {
  return finished_;
}

bool Scheduler::Store(DSState *ds_state) {
  std::unique_lock<std::mutex> lck(mutex_);
  ds_state->set_epochs(epochs_);
  for (auto &rparam : using_) {
    auto state = ds_state->add_states();
    state->set_begin(rparam->begin_);
    state->set_end(rparam->end_);
    state->set_epoch(rparam->epoch_);
    state->set_path(rparam->path_);
  }

  rparams_.Travel([ds_state](const ReadParam *rparam, size_t i) {
                  if (rparam == nullptr) {
                    return;
                  }
                  auto state = ds_state->add_states();
                  state->set_begin(rparam->begin_);
                  state->set_end(rparam->end_);
                  state->set_epoch(rparam->epoch_);
                  state->set_path(rparam->path_);
                  });
  XDL_LOG(DEBUG) << "schedule store " << ds_state->DebugString();
  return true;
}

bool Scheduler::Restore(const DSState &ds_state) {
  std::unique_lock<std::mutex> lck(mutex_);
  XDL_CHECK(rparams_.Empty());
  epochs_ = ds_state.epochs();
  XDL_CHECK(epochs_ < kEpochMax);
  size_t count = 0;
  for (int i = 0; i < ds_state.states_size(); ++i, ++count) {
    auto &state = ds_state.states(i);
    ReadParam *rparam = new ReadParam();
    rparam->path_ = strdup(state.path().c_str());
    rparam->epoch_ = state.epoch();
    XDL_CHECK(epochs_ == 0 || rparam->epoch_ < epochs_);
    rparam->ant_ = (rparam->epoch_ == 0 ? fs_->GetAnt(rparam->path_) : nullptr);
    rparam->begin_ = state.begin();
    rparam->end_ = state.end();
    XDL_CHECK(rparam->end_ > rparam->begin_);

    rparams_.Enqueue(rparam);
  }
  if (epochs_ != 0 || count == 0) {
    rparams_.Enqueue(nullptr);
    XDL_LOG(DEBUG) << "schedule restore nullptr as end";
  }
  restored_ = true;
  XDL_LOG(DEBUG) << "schedule restore " << ds_state.DebugString();
  return true;
}

}  // namespace io
}  // namespace xdl
