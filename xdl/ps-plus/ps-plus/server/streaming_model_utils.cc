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

#include "ps-plus/server/streaming_model_utils.h"

namespace ps {
namespace server {

Status StreamingModelUtils::WriteDense(const std::string& var) {
  std::unique_lock<std::mutex> lock(logger_.logger->mu);
  DenseLog& log = logger_.logger->dense[var];
  log.clear = false;
  return Status::Ok();
}

Status StreamingModelUtils::WriteSparse(const std::string& var, const Tensor& data) {
  std::unique_lock<std::mutex> lock(logger_.logger->mu);
  SparseLog& log = logger_.logger->sparse[var];
  if (data.Shape().Size() != 1) {
    return Status::ArgumentError("StreamingModelUtils WriteSparse Error: Shape Error");
  }
  CASES(data.Type(), {
    T* raw = data.Raw<T>();
    for (size_t i = 0; i < data.Shape()[0]; i++) {
      log.write_ids.insert(raw[i]);
    }
  });
  return Status::Ok();
}

Status StreamingModelUtils::WriteHash(const std::string& var, const Tensor& data) {
  std::unique_lock<std::mutex> lock(logger_.logger->mu);
  HashLog& log = logger_.logger->hash[var];
  if (data.Shape().Size() != 2 || data.Shape()[1] != 2) {
    return Status::ArgumentError("StreamingModelUtils WriteHash Error: Shape Error");
  }
  CASES(data.Type(), {
    T* raw = data.Raw<T>();
    for (size_t i = 0; i < data.Shape()[0]; i++) {
      log.write_ids.insert(std::pair<int64_t, int64_t>(raw[i * 2], raw[i * 2 + 1]));
    }
  });
  return Status::Ok();
}

Status StreamingModelUtils::DelHash(const std::string& var, const std::vector<int64_t>& data) {
  std::unique_lock<std::mutex> lock(logger_.logger->mu);
  HashLog& log = logger_.logger->hash[var];
  for (size_t i = 0; i < data.size() / 2; i++) {
    log.del_ids.insert(std::pair<int64_t, int64_t>(data[i * 2], data[i * 2 + 1]));
  }
  return Status::Ok();
}

Status StreamingModelUtils::GetDense(std::unordered_map<std::string, DenseLog>* result) {
  std::unique_lock<std::mutex> lock(mu_);
  result->clear();
  for (auto&& logger : loggers_) {
    {
      std::unique_lock<std::mutex> lock(logger->mu);
      std::swap(logger->dense, logger->dense_back);
    }
    auto& log = logger->dense_back;
    for (auto& item : log) {
      (*result)[item.first].Combine(item.second);
      item.second.Clear();
    }
  }
  std::vector<std::string> clears;
  for (auto item : *result) {
    if (item.second.IsClear()) {
      clears.push_back(item.first);
    }
  }
  for (auto item : clears) {
    result->erase(item);
  }
  return Status::Ok();
}

Status StreamingModelUtils::GetSparse(std::unordered_map<std::string, SparseLog>* result) {
  std::unique_lock<std::mutex> lock(mu_);
  result->clear();
  for (auto&& logger : loggers_) {
    {
      std::unique_lock<std::mutex> lock(logger->mu);
      std::swap(logger->sparse, logger->sparse_back);
    }
    auto& log = logger->sparse_back;
    for (auto& item : log) {
      (*result)[item.first].Combine(item.second);
      item.second.Clear();
    }
  }
  std::vector<std::string> clears;
  for (auto item : *result) {
    if (item.second.IsClear()) {
      clears.push_back(item.first);
    }
  }
  for (auto item : clears) {
    result->erase(item);
  }
  return Status::Ok();
}

Status StreamingModelUtils::GetHash(std::unordered_map<std::string, HashLog>* result) {
  std::unique_lock<std::mutex> lock(mu_);
  result->clear();
  for (auto&& logger : loggers_) {
    {
      std::unique_lock<std::mutex> lock(logger->mu);
      std::swap(logger->hash, logger->hash_back);
    }
    auto& log = logger->hash_back;
    for (auto& item : log) {
      (*result)[item.first].Combine(item.second);
      item.second.Clear();
    }
  }
  std::vector<std::string> clears;
  for (auto&& item : *result) {
    if (item.second.IsClear()) {
      clears.push_back(item.first);
    }
  }
  for (auto item : clears) {
    result->erase(item);
  }
  for (auto&& item : *result) {
    for (auto&& id : item.second.del_ids) {
      item.second.write_ids.erase(id);
    }
  }
  return Status::Ok();
}

StreamingModelUtils::LoggerRegister::LoggerRegister() {
  std::unique_lock<std::mutex> lock(mu_);
  logger.reset(new Logger);
  loggers_.insert(logger.get());
}

StreamingModelUtils::LoggerRegister::~LoggerRegister() {
  std::unique_lock<std::mutex> lock(mu_);
  loggers_.erase(logger.get());
}

std::mutex StreamingModelUtils::mu_;
std::unordered_set<StreamingModelUtils::Logger*> StreamingModelUtils::loggers_;
thread_local StreamingModelUtils::LoggerRegister StreamingModelUtils::logger_;

}
}

