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

#include "ps-plus/client/partitioner.h"

namespace ps {
namespace client {

PartitionerContext::PartitionerContext() {}

PartitionerContext::PartitionerContext(const VariableInfo& variable_info)
  : variable_info_(variable_info) {
  deleter_.resize(variable_info_.parts.size());
}

Data* PartitionerContext::GetData(size_t id) {
  if (id >= datas_.size()) {
    return nullptr;
  } else {
    return datas_[id].get();
  }
}

void PartitionerContext::SetData(size_t id, Data* data) {
  datas_.resize(std::max(id + 1, datas_.size()));
  datas_[id].reset(data);
}

void PartitionerContext::AddDeleter(Data* data) {
  deleter_[0].emplace_back(data);
}

void PartitionerContext::AddDeleter(Data* data, size_t index) {
  deleter_[index].emplace_back(data);
}

void PartitionerContext::SetVariableInfo(const VariableInfo& info) {
  variable_info_ = info;
  deleter_.resize(variable_info_.parts.size());
}

VariableInfo* PartitionerContext::GetVariableInfo() {
  return &variable_info_;
}

Status Partitioner::Split(PartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
  return Status::NotImplemented("Partitioner do not implement default Split");
}

Status Partitioner::Combine(PartitionerContext* ctx, Data* src, size_t server_id, std::unique_ptr<Data>* output) {
  return Status::NotImplemented("Partitioner do not implement default Combine");
}

Status Partitioner::Init(PartitionerContext* ctx, Data* src) {
  return Status::Ok();
}

Status Partitioner::CombineInit(PartitionerContext* ctx, std::unique_ptr<Data>* output) {
  return Status::Ok();
}

}
}

