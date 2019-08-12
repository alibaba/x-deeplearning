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

#include "ps-plus/client/partitioner/dense.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/initializer/none_initializer.h"
#include "ps-plus/common/thread_pool.h"
#include "xdl/core/utils/logging.h"
#include <cstring>

namespace ps {
namespace client {
namespace partitioner {

namespace {

Status SplitSingleTensor(PartitionerContext* ctx, const Tensor& data, std::vector<Data*>* dst) {
  VariableInfo* info = ctx->GetVariableInfo();
  if (info->parts.size() == 1 && info->shape.empty()) {
    dst->clear();
    WrapperData<Tensor>* src = new WrapperData<Tensor>(data);
    ctx->AddDeleter(src);
    dst->push_back(src);
    return Status::Ok();
  }
  if (data.Shape().IsScalar()) {
    return Status::ArgumentError("Scalar should be split by 1 part");
  }

  size_t parts_size = 0;
  for (auto item : info->parts) {
    parts_size += item.size;
  }
  if (data.Shape()[0] != parts_size) {
    return Status::ArgumentError("data's Shape is not match to VariableInfo");
  }
  dst->clear();

  TensorShape shape = data.Shape();
  DataType type = data.Type();
  char* ptr = data.Raw<char>();
  for (auto item : info->parts) {
    shape.Set(0, item.size);
    WrapperData<Tensor>* result = new WrapperData<Tensor>(type, shape, new initializer::NoneInitializer);
    size_t size = shape.NumElements() * SizeOfType(type);
    QuickMemcpy(result->Internal().Raw<char>(), ptr, size);
    ptr += size;
    ctx->AddDeleter(result);
    dst->push_back(result);
  }
  return Status::Ok();
}

}

Status Dense::Split(PartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
  VariableInfo* info = ctx->GetVariableInfo();
  if (info->type != VariableInfo::kIndex) {
    return Status::ArgumentError("Dense Partitioner Only Allow by kIndex");
  }
  if (dynamic_cast<WrapperData<Tensor>*>(src) != nullptr) {
    Tensor& data = dynamic_cast<WrapperData<Tensor>*>(src)->Internal();
    return SplitSingleTensor(ctx, data, dst);
  } else if (dynamic_cast<WrapperData<std::vector<Tensor>>*>(src) != nullptr) {
    std::vector<Tensor>& data_vec = dynamic_cast<WrapperData<std::vector<Tensor>>*>(src)->Internal();
    dst->clear();
    for (size_t i = 0; i < info->parts.size(); ++i) {
      WrapperData<std::vector<Tensor>>* result = new WrapperData<std::vector<Tensor>>();
      dst->emplace_back(result);
      ctx->AddDeleter(result);
    }
    for (size_t i = 0; i < data_vec.size(); ++i) {
      std::vector<Data*> one_dst;
      Status one_status = SplitSingleTensor(ctx, data_vec[i], &one_dst);
      if (!one_status.IsOk()) {
        return one_status;
      }
      for(size_t j = 0; j < one_dst.size(); ++j) {
        dynamic_cast<WrapperData<std::vector<Tensor>>*>((*dst)[j])->Internal().push_back(dynamic_cast<WrapperData<Tensor>*>(one_dst[j])->Internal());
      }
    }
    return Status::Ok();
  } else {
    return Status::ArgumentError("Dense Partitioner Only Allow the Tensor Data or Tensor Data Vector");
  }
}

Status Dense::Combine(PartitionerContext* ctx, Data* src, size_t server_id, std::unique_ptr<Data>* output) {
  VariableInfo* info = ctx->GetVariableInfo();
  if (info->type != VariableInfo::kIndex) {
    return Status::ArgumentError("Dense Partitioner Only Allow by kIndex");
  }

  WrapperData<Tensor>* raw_src = dynamic_cast<WrapperData<Tensor>*>(src);
  if (raw_src == nullptr) {
    return Status::ArgumentError("Dense Partitioner Combine src should be Tensor");
  }
  Tensor& data = raw_src->Internal();
  TensorShape shape = data.Shape();
  DataType type = data.Type();

  if (info->parts.size() != 1 || !info->shape.empty()) {
    if (shape.IsScalar()) {
      return Status::ArgumentError("Dense Partitioner Combiner src shape should not be scalar");
    }
    if (shape[0] != info->parts[server_id].size) {
      return Status::ArgumentError("Dense Partitioner Combiner src shape error");
    }
    size_t combined_size = 0;
    for (auto part : info->parts) {
      combined_size += part.size;
    }
    shape.Set(0, combined_size);
  }
  Tensor* result;
  /*
  if (output->get() == nullptr) {
    output->reset(new WrapperData<Tensor>(type, shape, new initializer::NoneInitializer));
    WrapperData<Tensor>* raw_output = dynamic_cast<WrapperData<Tensor>*>(output->get());
    result = &(raw_output->Internal());
  } else {
    WrapperData<Tensor>* raw_output = dynamic_cast<WrapperData<Tensor>*>(output->get());
    if (raw_output == nullptr) {
      return Status::ArgumentError("Dense Partitioner Combiner output is not Tensor");
    }
    result = &(raw_output->Internal());
  }
  */
  WrapperData<Tensor>* raw_output = dynamic_cast<WrapperData<Tensor>*>(output->get());
  if (raw_output == nullptr) {
    return Status::ArgumentError("Dense Partitioner Combiner output is not Tensor");
  }
  result = &(raw_output->Internal());
  size_t offset = 0;
  for (size_t i = 0; i < server_id; i++) {
    offset += info->parts[i].size;
  }
  if (!shape.IsScalar()) {
    offset *= shape.NumElements() / shape[0];
  }
  offset *= SizeOfType(type);
  size_t size = SizeOfType(type) * data.Shape().NumElements();
  QuickMemcpy(result->Raw<int8_t>() + offset, data.Raw<int8_t>(), size);
  return Status::Ok();
}

Status Dense::CombineInit(PartitionerContext* ctx, std::unique_ptr<Data>* output) {
  VariableInfo* info = ctx->GetVariableInfo();
  size_t combined_size = 0;
  for (auto part : info->parts) {
    combined_size += part.size;
  }
  std::vector<size_t> dims;
  if (!info->shape.empty()) {
    dims.push_back(combined_size);
    for (size_t i = 1; i < info->shape.size(); ++i) {
      dims.push_back(info->shape[i]);
    }
  }

  TensorShape shape(dims);
  DataType type = info->datatype;
  output->reset(new WrapperData<Tensor>(type, shape, new initializer::NoneInitializer));
  return Status::Ok();
}

}
}
}

