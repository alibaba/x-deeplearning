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

#ifndef PS_SCHEDULER_GLOBAL_FILE_QUEUE_H_
#define PS_SCHEDULER_GLOBAL_FILE_QUEUE_H_

#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <memory>
#include <set>
#include <thread>
#include <atomic>
#include <deque>
#include <unordered_map>

#include "ps-plus/common/status.h"
#include "ps-plus/message/worker_state.h"

namespace ps {

class GlobalFileQueue {
public:    
  GlobalFileQueue();
  ~GlobalFileQueue();
  Status Init(const std::vector<std::string>& paths,
              int epochs = 1, 
              bool epoch_isolate = false);
  Status GetNextFile(int worker_id, WorkerState* file);
  Status ReportWorkerState(
      int worker_id, 
      const std::vector<WorkerState>& worker_states);
  Status RestoreWorkerState(int worker_id);
  Status Serialize(std::string* buf);
  Status Deserialize(const std::string& buf);
  bool IsInitialized() const {
    std::unique_lock<std::mutex> lock(mu_);
    return initialized_;
  }

 private:
  bool IsAllWorkerFinishCurEpoch();
  struct FileInfo {
    size_t index_;
    size_t epoch_;
  };

  void SerializeFileInfos(
    const std::vector<std::vector<FileInfo> >& file_infos,
    std::string* buf);
  template <typename T>
  void SerializeWorkerStates(
      const std::vector<T>& worker_states,
      std::string* buf);
  size_t DeserializeFileInfos(
      char* buf,
      std::vector<std::vector<FileInfo> >* file_infos);
  template <typename T>  
  size_t DeserializeWorkerStates(
      char* buf,
      std::vector<T>* file_infos);  

 private:
  static const int MAX_WORKER_COUNT;
  mutable std::mutex mu_;
  std::vector<std::string> files_;
  std::unordered_map<std::string, size_t> file_index_;
  int epochs_;
  size_t cur_epoch_;
  size_t cur_file_index_;
  bool epoch_isolate_;
  bool initialized_;
  std::vector<std::vector<FileInfo> > used_files_;
  std::vector<std::vector<WorkerState> > last_report_;
  std::vector<std::deque<WorkerState> > restored_files_;
};

template <typename T>
void GlobalFileQueue::SerializeWorkerStates(
    const std::vector<T>& worker_states,
    std::string* buf) {
  size_t len = worker_states.size();
  buf->append((char*)&len, sizeof(size_t));    
  for (size_t i = 0; i < len; ++i) {
    auto& it = worker_states[i];
    size_t sub_len = it.size();
    buf->append((char*)&sub_len, sizeof(size_t));    
    for (auto& sub_it: it) {
      buf->append((char*)&sub_it.begin_, sizeof(size_t));        
      buf->append((char*)&sub_it.end_, sizeof(size_t));        
      buf->append((char*)&sub_it.epoch_, sizeof(size_t));        
      size_t path_len = sub_it.path_.size();
      buf->append((char*)&path_len, sizeof(size_t));            
      buf->append(sub_it.path_.data(), sub_it.path_.size());                
    }
  }
}

template <typename T>  
size_t GlobalFileQueue::DeserializeWorkerStates(
    char* base,
    std::vector<T>* worker_states) {
  worker_states->clear();
  char* ptr = base;
  size_t len = *(reinterpret_cast<size_t*>(ptr));
  ptr += sizeof(size_t);
  for (size_t i = 0; i < len; ++i) {
    worker_states->emplace_back();
    size_t sub_len = *(reinterpret_cast<size_t*>(ptr));
    ptr += sizeof(size_t);
    for (size_t j = 0; j < sub_len; ++j) {
      WorkerState worker_state;
      worker_state.begin_ = *(reinterpret_cast<size_t*>(ptr));
      ptr += sizeof(size_t);
      worker_state.end_ = *(reinterpret_cast<size_t*>(ptr));
      ptr += sizeof(size_t);
      worker_state.epoch_ = *(reinterpret_cast<size_t*>(ptr));
      ptr += sizeof(size_t);
      size_t path_len = *(reinterpret_cast<size_t*>(ptr));
      ptr += sizeof(size_t);
      worker_state.path_.assign(ptr, path_len);
      ptr += path_len;
      (*worker_states)[i].push_back(worker_state);
    }
  }

  return ptr - base;
}


} // namespace ps

#endif // PS_SCHEDULER_GLOBAL_FILE_QUEUE_H_
