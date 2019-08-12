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

#include "ps-plus/client/partitioner/merged_hash.h"
#include "ps-plus/common/thread_pool.h"
#include "ps-plus/common/hasher.h"
#include "ps-plus/common/logging.h"
#include <cstring>
#include <iostream>
#include <future>

namespace ps {
namespace client {
namespace partitioner {

Status MergedHashData::Split(MergedPartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
  WrapperData<std::vector<Tensor>>* data_wrapper = dynamic_cast<WrapperData<std::vector<Tensor>>*>(src);
  if (data_wrapper == nullptr) {
    return Status::ArgumentError("MergedHashData Partitioner Only Allow the Tensor Data Vector");
  }
  std::vector<Tensor>& data_vec = data_wrapper->Internal();
  if (data_vec.size() != ctx->ContextSize()) {
    return Status::ArgumentError("Merged Variable Number Should be the Same With Data Number");
  }
  size_t part_size = ctx->GetContext(0)->GetVariableInfo()->parts.size();
  dst->clear();
  for (size_t i = 0; i < part_size; ++i) {
    WrapperData<std::vector<Tensor>>* result = new WrapperData<std::vector<Tensor>>();
    dynamic_cast<WrapperData<std::vector<Tensor>>*>(result)->Internal().resize(ctx->ContextSize());
    dst->emplace_back(result);
    ctx->AddDeleter(result);
  }
  Status status = Status::Ok();
  std::vector<std::vector<Data*>> dst_arr(ctx->ContextSize());
  for (size_t i = 0; i < ctx->ContextSize(); ++i) {
    WrapperData<Tensor>* one_src = new WrapperData<Tensor>(data_vec[i]);
    ctx->GetContext(i)->AddDeleter(one_src);
    Status one_status = data_partitioner_.Split(ctx->GetContext(i), one_src, &dst_arr[i]);
    if (!one_status.IsOk()) {
      status = one_status;
    }
    for(size_t j = 0; j < dst_arr[i].size(); ++j) {
      dynamic_cast<WrapperData<std::vector<Tensor>>*>((*dst)[j])->Internal()[i] = dynamic_cast<WrapperData<Tensor>*>(dst_arr[i][j])->Internal();
    }
  }
  return status;
}

Status MergedHashData::Combine(MergedPartitionerContext* ctx, Data* src, size_t server_id, std::vector<std::unique_ptr<Data>>* output) {
  WrapperData<std::vector<Tensor>>* data_wrapper = dynamic_cast<WrapperData<std::vector<Tensor>>*>(src);
  if (data_wrapper == nullptr) {
    return Status::ArgumentError("MergedHashData Partitioner Combine src Only Allow the Tensor Data Vector");
  }
  std::vector<Tensor>& data_vec = data_wrapper->Internal();
  if (data_vec.size() != ctx->ContextSize() || data_vec.size() != output->size()) {
    return Status::ArgumentError("Merged Variable Number Should be the Same With Data Number");
  }
  Status status = Status::Ok();
  for (size_t i = 0; i < ctx->ContextSize(); ++i) {
    WrapperData<Tensor>* one_src = new WrapperData<Tensor>(data_vec[i]);
    ctx->GetContext(i)->AddDeleter(one_src, server_id);
    Status one_status = data_partitioner_.Combine(ctx->GetContext(i), one_src, server_id, &(*output)[i]);
    if (!one_status.IsOk()) {
      status = one_status;
    }
  }
  return status;
}

Status MergedHashId::Init(MergedPartitionerContext* ctx, Data* src) {
  WrapperData<std::vector<Tensor>>* data_wrapper = dynamic_cast<WrapperData<std::vector<Tensor>>*>(src);
  if (data_wrapper == nullptr) {
    return Status::ArgumentError("MergedHashId Partitioner Only Allow the Tensor Data Vector");
  }
  std::vector<Tensor>& id_vec = data_wrapper->Internal();
  if (id_vec.size() != ctx->ContextSize()) {
    return Status::ArgumentError("Merged Variable Number Should be the Same With Id Number");
  }
  size_t part_size = ctx->GetContext(0)->GetVariableInfo()->parts.size();
  return MultiThreadDo(ctx->ContextSize(), [&](const Range& r) {
        for (size_t i = r.begin; i < r.end; i++) {
          if (ctx->GetContext(i)->GetVariableInfo()->parts.size() != part_size) {
            return Status::ArgumentError("Merged Hash Variable Should Have the Same Parts Size");
          }
          if (ctx->GetContext(i)->GetVariableInfo()->type != VariableInfo::kHash128 && 
              ctx->GetContext(i)->GetVariableInfo()->type != VariableInfo::kHash64) {
            return Status::ArgumentError("HashId Partitioner Only Allow by kHash");
          }
          WrapperData<Tensor>* one_src = new WrapperData<Tensor>(id_vec[i]);
          ctx->GetContext(i)->AddDeleter(one_src);
          Status st = id_partitioner_.Init(ctx->GetContext(i), one_src);
          if (!st.IsOk()) {
            return st;
          }
        }
        return Status::Ok();
      }, 1);
}

Status MergedHashData::CombineInit(MergedPartitionerContext* ctx, std::vector<std::unique_ptr<Data>>* output) {
  Status status = Status::Ok();
  for (size_t i = 0; i < ctx->ContextSize(); ++i) {
    Status one_status = data_partitioner_.CombineInit(ctx->GetContext(i), &(*output)[i]);
    if (!one_status.IsOk()) {
      status = one_status;
    }
  }
  return status;
}

}
}
}

