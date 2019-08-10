/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "ps-plus/common/hdfs_data_source.h"

#include <thread>
#include <memory>
#include <cstring>
#include <atomic>
#include "ps-plus/common/file_system.h"
#include <sstream>
#include <sys/time.h>

namespace ps {

namespace {
void LOG_TIME(std::string name) {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  std::cout << name + ":" + std::to_string(1000000 * tv.tv_sec + tv.tv_usec) + "\n";
}
}

Status HdfsDataSource::Init(int rank, int worker_num, 
                          size_t default_value_length) {
  default_value_length_ = default_value_length;

  std::vector<std::thread> threads;

  size_t size = file_num_ / worker_num + file_num_ % worker_num < rank;
  std::atomic<size_t> counter(0);
  for (size_t p = 0; p < 16; p++) {
    threads.emplace_back([this, p, worker_num, rank, size, &counter] {
      size_t beg = file_num_ / 16 * p, end = p == 15 ? file_num_ : file_num_ / 16 * (p + 1);
      for (size_t k = beg; k < end; ++k) {
        if (k % worker_num == rank) {
          Status st = InitFromFile(filepath_ + "." + std::to_string(k));
          if (!st.IsOk()) {
            std::unique_lock<std::mutex> lock(mu_);
            if (status_.IsOk()) {
              status_ = st;
            }
          }
          std::cout << "Read 1 file for datasource " + std::to_string(++counter) + "/" + std::to_string(size) + " done\n";
        }
      }
    });
  }
  for (auto&& thread : threads) {
    thread.join();
  }
  return status_;
}

Status HdfsDataSource::InitFromFile(const std::string& filepath) {
  int count;
  char* ptr;
  {
    std::unique_lock<std::mutex> lock(mu_);
    count = kBufferCount;
    ptr = new char[default_value_length_ * kBufferCount];
    buffer_.emplace_back(ptr);
  }
  FileSystem* fs = nullptr;
  PS_CHECK_STATUS(FileSystem::GetFileSystem(filepath, &fs));
  FileSystem::ReadStream* stream = nullptr;
  PS_CHECK_STATUS(fs->OpenReadStream(filepath, &stream));
  int64_t id;
  void* value = nullptr;
  std::vector<std::pair<int64_t, char*>> datas;
  while (stream->Read(&id, sizeof(id)) == Status::Ok()) {
    std::unique_ptr<char[]> value(new char[default_value_length_]);
    PS_CHECK_STATUS(stream->Read(ptr, default_value_length_));
    datas.emplace_back(id, ptr);
    ptr += default_value_length_;
    if (--count == 0) {
      std::unique_lock<std::mutex> lock(mu_);
      count = kBufferCount;
      ptr = new char[default_value_length_ * kBufferCount];
      buffer_.emplace_back(ptr);
    }
  }
  std::unique_lock<std::mutex> lock(mu_);
  for (auto& item : datas) {
    data_[item.first] = item.second;
  }
  return Status::Ok();
}

Status HdfsDataSource::Get(int64_t id, DataClosure* closure) {
  auto iter = data_.find(id);
  if (iter == data_.end()) {
    return Status::NotFound("DataSource " + std::to_string(id) + " not found");
  }
  closure->data = iter->second;
  closure->length = default_value_length_;
  return Status::Ok();
}

Status HdfsDataSource::BatchGet(const std::vector<int64_t>& ids,
                                std::vector<DataClosure>* closures) {
  closures->clear();
  for (auto id : ids) {
    closures->emplace_back();
    PS_CHECK_STATUS(Get(id, &closures->back()));
  }
  return Status::Ok();
}

void HdfsDataSource::BatchGetV2(const std::vector<int64_t>& ids,
                                std::vector<DataClosure>* closures,
                                std::vector<int64_t>* rst_ids) {
  closures->clear();
  for (size_t i = 0; i < ids.size(); i++) {
    auto iter = data_.find(ids[i]);
    if (iter == data_.end()) {
      continue;
    }
    closures->emplace_back();
    DataClosure* closure = &closures->back();
    closure->data = iter->second;
    closure->length = default_value_length_;
    rst_ids->push_back(i);
  }
}

}  // namespace ps

