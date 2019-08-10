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

#ifndef PS_PLUS_CLIENT_PARTITIONER_MERGED_SLICE_H_
#define PS_PLUS_CLIENT_PARTITIONER_MERGED_SLICE_H_

#include "ps-plus/client/merged_partitioner.h"
#include "ps-plus/client/partitioner/sparse.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/tensor_shape.h"

namespace ps {
namespace client {
namespace partitioner {

class MergedSliceBegin : public MergedPartitioner {
 public:
  MergedSliceBegin() {}
  virtual Status Split(MergedPartitionerContext* ctx, Data* src, std::vector<Data*>* dst) override {
    WrapperData<std::vector<int>>* data_wrapper = dynamic_cast<WrapperData<std::vector<int>>*>(src);
    if (data_wrapper == nullptr) {
      return Status::ArgumentError("MergedSliceBegin Partitioner Only Allow the IntVector");
    }
    auto& data = data_wrapper->Internal();
    if (data.size() != ctx->ContextSize()) {
      return Status::ArgumentError("MergedSliceBegin Partitioner IntVector Size should be equal to variable size");
    }
    std::vector<std::vector<int>> raw_result(ctx->GetContext(0)->GetVariableInfo()->parts.size());
    for (size_t i = 0; i < ctx->ContextSize(); i++) {
      VariableInfo* info = ctx->GetContext(0)->GetVariableInfo();
      if (info->parts.size() != raw_result.size()) {
        return Status::ArgumentError("Merged Variable Number Should be the Same With Data Number");
      }
      int k = 0;
      for (size_t j = 0; j < info->parts.size(); j++) {
        raw_result[j].push_back(k);
        k += info->parts[j].size;
      }
    }
    for (auto&& item : raw_result) {
      auto rst = new WrapperData<std::vector<int>>(item);
      dst->push_back(rst);
      ctx->AddDeleter(rst);
    }
    return Status::Ok();
  }
};

class MergedSliceEnd : public MergedPartitioner {
 public:
  MergedSliceEnd() {}
  virtual Status Split(MergedPartitionerContext* ctx, Data* src, std::vector<Data*>* dst) override {
    WrapperData<std::vector<int>>* data_wrapper = dynamic_cast<WrapperData<std::vector<int>>*>(src);
    if (data_wrapper == nullptr) {
      return Status::ArgumentError("MergedSliceBegin Partitioner Only Allow the IntVector");
    }
    auto& data = data_wrapper->Internal();
    if (data.size() != ctx->ContextSize()) {
      return Status::ArgumentError("MergedSliceBegin Partitioner IntVector Size should be equal to variable size");
    }
    std::vector<std::vector<int>> raw_result(ctx->GetContext(0)->GetVariableInfo()->parts.size());
    for (size_t i = 0; i < ctx->ContextSize(); i++) {
      VariableInfo* info = ctx->GetContext(0)->GetVariableInfo();
      if (info->parts.size() != raw_result.size()) {
        return Status::ArgumentError("Merged Variable Number Should be the Same With Data Number");
      }
      int k = 0;
      for (size_t j = 0; j < info->parts.size(); j++) {
        k += info->parts[j].size;
        raw_result[j].push_back(k);
      }
    }
    for (auto&& item : raw_result) {
      auto rst = new WrapperData<std::vector<int>>(item);
      dst->push_back(rst);
      ctx->AddDeleter(rst);
    }
    return Status::Ok();
  }
};

}
}
}

#endif

