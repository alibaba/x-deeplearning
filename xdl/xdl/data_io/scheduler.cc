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
#include "xdl/core/utils/logging.h"

#include <string.h>

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
  epochs_ = epochs;
  return true;
}

bool Scheduler::SetShuffle(bool shuffle) {
  shuffle_ = shuffle;
  return true;
}

bool Scheduler::SetZType(ZType ztype) {
  ztype_ = ztype;
  return true;
}

bool Scheduler::Schedule() {
  std::sort(paths_.begin(), paths_.end());
  finished_ = false;

  std::unique_lock<std::mutex> lck(mutex_);
  if (restored_) {
    XDL_LOG(DEBUG) << "Has been restored, no need to scheduled again";
    return false;
  }
  if (rparams_.Size() > 0 || using_.size() > 0) {
    XDL_LOG(DEBUG) << "Already scheduled, size=" << using_.size() << " + "<< rparams_.Size();
    return false;
  }

  for (unsigned i = 0; i < paths_.size(); ++i) {
    XDL_LOG(DEBUG) << "{" << i << "} " << paths_[i];
  }

  std::vector<ReadParam *>rparams;
  for (unsigned i = 0; i < paths_.size(); ++i) {
    ReadParam *rparam = new ReadParam();
    rparam->pathid_ = i;
    rparam->path_ = paths_[i].c_str();
    rparam->epoch_ = 0;
    rparam->ant_ = nullptr;
    rparam->begin_ = 0;
    rparam->end_ = 0;
    rparam->parsed_ = 0;
    rparams.push_back(rparam);
  }

  if (rparams.size() == 0) {
    finished_ = true;
    return false;
  }

  if (shuffle_) {
    std::random_shuffle(rparams.begin(), rparams.end());
  }

  for (auto rparam : rparams) {
    rparams_.Enqueue(rparam);
    XDL_LOG(DEBUG) << "schedule " << rparam->DebugString();
  }

  return true;
}

ReadParam *Scheduler::Acquire() {
  ReadParam *rparam =  nullptr;
  while (!finished_) {
    if (rparams_.TryDequeue(&rparam, 1000)) {
      break;
    }
  }

  if (rparam == nullptr) {
    finished_ = true;
    return nullptr;
  }
  if (rparam->ant_ == nullptr) {
    rparam->ant_ = fs_->GetZAnt(rparam->path_, ztype_);
    rparam->ant_->Seek(rparam->begin_);
  }
  if (rparam->end_ == 0) {
    if (ztype_ == kZLib) {
      rparam->end_ = ULONG_MAX;
    } else {
      rparam->end_ = fs_->Size(rparam->path_);
    }
    XDL_CHECK(rparam->end_ > rparam->begin_);
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
  if (epochs_ == 0 || rparam->epoch_ < epochs_ - 1) {
    ++ rparam->epoch_;
    rparam->begin_ = 0;
    rparam->parsed_ = 0;
    XDL_CHECK(rparam->ant_ != nullptr);
    rparam->ant_->Seek(0);
    rparams_.Enqueue(rparam);
    XDL_LOG(DEBUG) << "re schedule " << rparam->DebugString();
  } else {
    /* insert nullptr while all paths done */
    XDL_LOG(DEBUG) << "finish " << rparam->DebugString();
    if (using_.empty()) {
      rparams_.Enqueue(nullptr);
      XDL_LOG(DEBUG) << "finish all, schedule nullptr as end";
    }
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
    state->set_begin(rparam->parsed_);
    state->set_end(rparam->end_);
    state->set_epoch(rparam->epoch_);
    state->set_pathid(rparam->pathid_);
  }

  rparams_.Travel([ds_state, this](const ReadParam *rparam, size_t i) {
                  if (rparam == nullptr) {
                    return;
                  }
                  auto state = ds_state->add_states();
                  state->set_begin(rparam->parsed_);
                  state->set_end(rparam->end_);
                  state->set_epoch(rparam->epoch_);
                  XDL_CHECK(rparam->pathid_ < this->paths_.size());
                  state->set_pathid(rparam->pathid_);
                  });
  XDL_LOG(DEBUG) << "schedule store " << ds_state->DebugString();

  return true;
}

bool Scheduler::Restore(const DSState &ds_state) {
  Clear();
  std::unique_lock<std::mutex> lck(mutex_);
  epochs_ = ds_state.epochs();
  size_t count = 0;
  for (int i = 0; i < ds_state.states_size(); ++i, ++count) {
    auto &state = ds_state.states(i);
    ReadParam *rparam = new ReadParam();
    rparam->pathid_ = state.pathid();
    XDL_CHECK(rparam->pathid_ < paths_.size()) << "pathid=" << rparam->pathid_ << " size=" << paths_.size();
    rparam->path_ = paths_[rparam->pathid_].c_str();
    rparam->epoch_ = state.epoch();
    XDL_CHECK(epochs_ == 0 || rparam->epoch_ < epochs_);
    rparam->ant_ = nullptr;
    rparam->parsed_ = rparam->begin_ = state.begin();
    rparam->end_ = state.end();
    XDL_CHECK(rparam->end_ >= rparam->begin_);

    rparams_.Enqueue(rparam);
  }
  restored_ = true;
  XDL_LOG(DEBUG) << "schedule restore " << ds_state.DebugString();
  if (rparams_.Size() == 0) {
    finished_ = true;
    return false;
  }
  return true;
}

void Scheduler::Clear() {
  std::unique_lock<std::mutex> lck(mutex_);
  //TODO: free
  rparams_.Clear();
  using_.clear();
}

}  // namespace io
}  // namespace xdl
