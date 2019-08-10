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

#include "ps-plus/client/merged_partitioner.h"

namespace ps {
namespace client {

MergedPartitionerContext::MergedPartitionerContext() {}

Data* MergedPartitionerContext::GetData(std::size_t id) {
  if (id >= datas_.size()) {
    return nullptr;
  } else {
    return datas_[id].get();
  }
}

void MergedPartitionerContext::SetData(std::size_t id, Data* data) {
  datas_.resize(std::max(id + 1, datas_.size()));
  datas_[id].reset(data);
}

void MergedPartitionerContext::AddDeleter(Data* data) {
  deleter_.emplace_back(data);
}

void MergedPartitionerContext::SetContext(std::size_t id, PartitionerContext* context) {
  context_.resize(std::max(id + 1, context_.size()));
  context_[id].reset(context);
}

PartitionerContext* MergedPartitionerContext::GetContext(std::size_t id) {
  if (id >= context_.size()) {
    return nullptr;
  } else {
    return context_[id].get();
  } 
}

void MergedPartitionerContext::AddContext(PartitionerContext* context) {
  context_.emplace_back(context);
}

size_t MergedPartitionerContext::ContextSize() {
  return context_.size();
}

Status MergedPartitioner::Split(MergedPartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
  return Status::NotImplemented("MergedPartitioner do not implement default Split");
}

Status MergedPartitioner::Combine(MergedPartitionerContext* ctx, Data* src, size_t server_id, std::vector<std::unique_ptr<Data>>* output) {
  return Status::NotImplemented("MergedPartitioner do not implement default Combine");
}

Status MergedPartitioner::Init(MergedPartitionerContext* ctx, Data* src) {
  return Status::Ok();
}

Status MergedPartitioner::CombineInit(MergedPartitionerContext* ctx, std::vector<std::unique_ptr<Data>>* output) {
  return Status::Ok();
}

}
}

