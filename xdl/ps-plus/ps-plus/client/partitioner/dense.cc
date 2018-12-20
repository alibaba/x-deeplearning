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
#include <cstring>

namespace ps {
namespace client {
namespace partitioner {

Status Dense::Split(PartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
  VariableInfo* info = ctx->GetVariableInfo();
  if (info->type != VariableInfo::kIndex) {
    return Status::ArgumentError("Dense Partitioner Only Allow by kIndex");
  }
  WrapperData<Tensor>* data_wrapper = dynamic_cast<WrapperData<Tensor>*>(src);
  if (data_wrapper == nullptr) {
    return Status::ArgumentError("Dense Partitioner Only Allow the Tensor Data");
  }
  Tensor& data = data_wrapper->Internal();

  if (info->parts.size() == 1 && info->shape.empty()) {
    dst->clear();
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

}
}
}

