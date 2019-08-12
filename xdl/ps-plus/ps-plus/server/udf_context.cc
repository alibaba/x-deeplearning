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

#include "ps-plus/server/udf_context.h"

#include <algorithm>

namespace ps {
namespace server {

UdfContext::UdfContext()
  : variable_(nullptr), storage_manager_(nullptr), locker_(nullptr), server_locker_(nullptr), streaming_model_args_(nullptr) {
}

UdfContext::~UdfContext() {
  for (auto dep : dependencies_) {
    delete dep;
  }
}

size_t UdfContext::DataSize() {
  return datas_.size();
}

Status UdfContext::ProcessOutputs(const std::vector<size_t>& output_ids) {
  outputs_.clear();
  for (auto id : output_ids) {
    if (id >= datas_.size()) {
      outputs_.clear();
      return Status::IndexOverflow("UdfContext Process Output Id Overflow");
    }
    outputs_.push_back(datas_[id]);
  }
  return Status::Ok();
}

const std::vector<Data*>& UdfContext::Outputs() {
  return outputs_;
}

Status UdfContext::SetStorageManager(StorageManager* storage_manager) {
  storage_manager_ = storage_manager;
  return Status::Ok();
}

Status UdfContext::SetVariable(Variable* variable) {
  variable_ = variable;
  return Status::Ok();
}

Status UdfContext::SetVariableName(const std::string& variable_name) {
  variable_name_ = variable_name;
  return Status::Ok();
}

Status UdfContext::SetLocker(QRWLocker* locker) {
  locker_ = locker;
  return Status::Ok();
}

Status UdfContext::SetServerLocker(QRWLocker* locker) {
  server_locker_ = locker;
  return Status::Ok();
}

Status UdfContext::SetStreamingModelArgs(StreamingModelArgs* streaming_model_args) {
  streaming_model_args_ = streaming_model_args;
  return Status::Ok();
}

Status UdfContext::SetData(size_t id, Data* data, bool need_free) {
  datas_.resize(std::max(id + 1, datas_.size()), nullptr);
  datas_[id] = data;
  if (need_free) {
    dependencies_.push_back(data);
  }
  return Status::Ok();
}

Status UdfContext::GetData(size_t id, Data** data) {
  datas_.resize(std::max(id + 1, datas_.size()), nullptr);
  *data = datas_[id];
  return Status::Ok();
}

Status UdfContext::AddDependency(Data* dependency) {
  dependencies_.push_back(dependency);
  return Status::Ok();
}

void UdfContext::RemoveOutputDependency() {
  std::vector<Data*> dependencies;
  for (auto dep : dependencies_) {
    bool remove = false;
    for (auto x : outputs_) {
      if (x == dep) {
        remove = true;
      }
    }
    if (!remove) {
      dependencies.push_back(dep);
    }
  }
  dependencies_ = std::move(dependencies);
}

StorageManager* UdfContext::GetStorageManager() {
  return storage_manager_;
}

Variable* UdfContext::GetVariable() {
  return variable_;
}

const std::string& UdfContext::GetVariableName() {
  return variable_name_;
}

QRWLocker* UdfContext::GetLocker() {
  return locker_;
}

QRWLocker* UdfContext::GetServerLocker() {
  return server_locker_;
}

StreamingModelArgs* UdfContext::GetStreamingModelArgs() {
  return streaming_model_args_;
}

}
}

