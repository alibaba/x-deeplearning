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

#include "global_file_queue.h"

#include "ps-plus/common/logging.h"
#include "ps-plus/common/status.h"

#include <string>
#include <iostream>

using namespace std;
using namespace std::chrono;
using namespace ps;

namespace ps {

const int GlobalFileQueue::MAX_WORKER_COUNT = 10000;

GlobalFileQueue::GlobalFileQueue() 
  : epochs_(1)
  , cur_epoch_(0)
  , cur_file_index_(0)
  , epoch_isolate_(false)
  , initialized_(false) {
}

GlobalFileQueue::~GlobalFileQueue() {
}

Status GlobalFileQueue::Init(
    const std::vector<std::string>& paths,
    int epochs, 
    bool epoch_isolate) {
  std::unique_lock<std::mutex> lock(mu_);
  if (!initialized_) {
    epochs_ = epochs;
    epoch_isolate_ = epoch_isolate;
    files_ = paths;
    used_files_.resize(MAX_WORKER_COUNT);
    restored_files_.resize(MAX_WORKER_COUNT);
    last_report_.resize(MAX_WORKER_COUNT);
    for (size_t i = 0; i < files_.size(); ++i) {
      file_index_[files_[i]] = i;
    }

    initialized_ = true;
  }

  return Status::Ok();
}

Status GlobalFileQueue::GetNextFile(
    int worker_id , 
    WorkerState* file) {
  std::unique_lock<std::mutex> lock(mu_);
  if (!initialized_) {
    return Status::FileQueueNeedWait("not initialized");
  } 

  std::deque<WorkerState>& restored_files = restored_files_[worker_id];
  if (!restored_files.empty()) {
    *file = restored_files.front();
    restored_files.pop_front();
    used_files_[worker_id].push_back(
        FileInfo{.index_ = file_index_[file->path_],
            .epoch_ = file->epoch_});
  } else {
    if (cur_file_index_ >= files_.size()) {
      if (cur_epoch_ + 1 == epochs_) {
        file->path_ = "";
        return Status::Ok();
      }

      if (epoch_isolate_ && !IsAllWorkerFinishCurEpoch()) {
        return Status::FileQueueNeedWait("not all worker finish current epoch");
      } 
      
      ++cur_epoch_;
      cur_file_index_ = 0;
    } 

    file->path_ = files_[cur_file_index_];
    file->begin_ = 0;
    file->end_ = 0;
    file->epoch_ = cur_epoch_;
    used_files_[worker_id].push_back(
        FileInfo{.index_ = cur_file_index_,
            .epoch_ = cur_epoch_});
    ++cur_file_index_;
  }

  return Status::Ok();
}

bool GlobalFileQueue::IsAllWorkerFinishCurEpoch() {
  return true;
}

Status GlobalFileQueue::ReportWorkerState(
    int worker_id, 
    const std::vector<ps::WorkerState>& worker_states) {
  if (worker_id >= MAX_WORKER_COUNT) {
    return Status::ArgumentError("worker_id exceed MAX_WORKER_COUNT:10000");
  }

  std::unique_lock<std::mutex> lock(mu_);
  last_report_[worker_id] = worker_states;
  return Status::Ok();
}

Status GlobalFileQueue::RestoreWorkerState(
    int worker_id) {
  std::set<std::pair<size_t, size_t> > last_report;
  for (auto& worker_state: last_report_[worker_id]) {
    last_report.insert({file_index_[worker_state.path_], 
        worker_state.epoch_});
  }

  std::unique_lock<std::mutex> lock(mu_);
  std::deque<WorkerState>& restored_files = 
    restored_files_[worker_id];
  restored_files.clear();  
  restored_files.insert(
      restored_files.end(), 
      last_report_[worker_id].begin(),
      last_report_[worker_id].end());
  for (auto it = used_files_[worker_id].rbegin(); 
       it != used_files_[worker_id].rend(); 
       ++it) {
    if (last_report.find({it->index_, it->epoch_}) == 
        last_report.end()) {
      WorkerState ws;
      ws.path_ = files_[it->index_];
      ws.begin_ = 0;
      ws.end_ = 0;
      ws.epoch_ = it->epoch_;
      restored_files.push_back(ws);      
    } else {
      break;
    }
  }

  used_files_[worker_id].clear();
  return Status::Ok();
}

Status GlobalFileQueue::Serialize(std::string* buf) {
  unique_lock<std::mutex> lock(mu_);
  buf->append((char*)&cur_epoch_, sizeof(size_t));
  buf->append((char*)&cur_file_index_, sizeof(size_t));  
  SerializeFileInfos(used_files_, buf);
  SerializeWorkerStates<std::vector<WorkerState> >(
      last_report_, buf);
  SerializeWorkerStates<std::deque<WorkerState> >(
      restored_files_, buf);
  return Status::Ok();
}

void GlobalFileQueue::SerializeFileInfos(
    const std::vector<std::vector<FileInfo> >& file_infos,
    std::string* buf) {
  size_t len = file_infos.size();
  buf->append((char*)&len, sizeof(size_t));    
  for (size_t i = 0; i < len; ++i) {
    auto& it = file_infos[i];
    size_t sub_len = it.size();
    buf->append((char*)&sub_len, sizeof(size_t));    
    for (size_t j = 0; j < sub_len; ++j) {
      auto& sub_it = it[j];
      buf->append((char*)&sub_it.index_, sizeof(size_t));        
      buf->append((char*)&sub_it.epoch_, sizeof(size_t));        
    }
  }
}

Status GlobalFileQueue::Deserialize(const std::string& buf) {
  unique_lock<std::mutex> lock(mu_);
  char* ptr = const_cast<char*>(buf.data());
  cur_epoch_ = *(reinterpret_cast<size_t*>(ptr));  
  ptr += sizeof(size_t);
  cur_file_index_ = *(reinterpret_cast<size_t*>(ptr));    
  ptr += sizeof(size_t);
  ptr += DeserializeFileInfos(ptr, &used_files_);
  ptr += DeserializeWorkerStates<std::vector<WorkerState> >(
      ptr, &last_report_);
  ptr += DeserializeWorkerStates<std::deque<WorkerState> >(
      ptr, &restored_files_);
  if (ptr - buf.data() == buf.size()) {
    return Status::Ok();
  }

  return Status::DataLoss("global_file_queue deserialize error");
}

size_t GlobalFileQueue::DeserializeFileInfos(
      char* base,
      std::vector<std::vector<FileInfo> >* file_infos) {
  file_infos->clear();
  char* ptr = base;
  size_t len = *(reinterpret_cast<size_t*>(ptr));
  ptr += sizeof(size_t);
  for (size_t i = 0; i < len; ++i) {
    file_infos->emplace_back();
    size_t sub_len = *(reinterpret_cast<size_t*>(ptr));
    ptr += sizeof(size_t);
    for (size_t j = 0; j < sub_len; ++j) {
      FileInfo file_info;
      file_info.index_ = *(reinterpret_cast<size_t*>(ptr));
      ptr += sizeof(size_t);
      file_info.epoch_ = *(reinterpret_cast<size_t*>(ptr));
      ptr += sizeof(size_t);
      (*file_infos)[i].push_back(file_info);
    }
  }

  return ptr - base;
}

} // namespace ps
