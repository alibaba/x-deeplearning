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

#include "ps-plus/client/partitioner/sparse.h"
#include "ps-plus/common/initializer/none_initializer.h"
#include "ps-plus/common/hasher.h"
#include <cstring>
#include <iostream>

namespace ps {
namespace client {
namespace partitioner {

namespace {

struct SparseSlices {
  std::vector<std::vector<size_t>> ids;
  size_t id_size;
};

}

Status SparseData::Split(PartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
  WrapperData<Tensor>* data_wrapper = dynamic_cast<WrapperData<Tensor>*>(src);
  if (data_wrapper == nullptr) {
    return Status::ArgumentError("Sparse Partitioner Only Allow the Tensor Data");
  }
  Tensor& data = data_wrapper->Internal();

  if (data.Shape().IsScalar()) {
    return Status::ArgumentError("Sparse Partitioner Doesn't Accept Scalar Type");
  }

  WrapperData<SparseSlices>* id_wrapper = dynamic_cast<WrapperData<SparseSlices>*>(ctx->GetData(id_));
  if (id_wrapper == nullptr) {
    return Status::ArgumentError("Sparse Partitioner Init Error");
  }
  SparseSlices& slices = id_wrapper->Internal();

  if (data.Shape()[0] != slices.id_size) {
    return Status::ArgumentError("Data's Shape Does Not Match the ID Size");
  }

  dst->clear();

  TensorShape shape = data.Shape();
  DataType type = data.Type();
  shape.Set(0, 1);
  size_t single_size = shape.NumElements() * SizeOfType(type);
  for (auto item : slices.ids) {
    shape.Set(0, item.size());
    WrapperData<Tensor>* result = new WrapperData<Tensor>(type, shape, new initializer::NoneInitializer);
    char* res_ptr = result->Internal().Raw<char>();
    char* src_ptr = data.Raw<char>();
    for (auto id : item) {
      memcpy(res_ptr, src_ptr + id * single_size, single_size);
      res_ptr += single_size;
    }
    ctx->AddDeleter(result);
    dst->push_back(result);
  }
  return Status::Ok();
}

Status SparseData::Combine(PartitionerContext* ctx, Data* src, size_t server_id, std::unique_ptr<Data>* output) {
  WrapperData<SparseSlices>* id_wrapper = dynamic_cast<WrapperData<SparseSlices>*>(ctx->GetData(id_));
  if (id_wrapper == nullptr) {
    return Status::ArgumentError("Sparse Partitioner Init Error");
  }
  SparseSlices& slices = id_wrapper->Internal();

  WrapperData<Tensor>* raw_src = dynamic_cast<WrapperData<Tensor>*>(src);
  if (raw_src == nullptr) {
    return Status::ArgumentError("Sparse Partitioner Combine src should be Tensor");
  }
  Tensor& data = raw_src->Internal();
  TensorShape shape = data.Shape();
  DataType type = data.Type();
  if (shape.IsScalar()) {
    return Status::ArgumentError("Sparse Partitioner Combiner src shape should not be scalar");
  }
  if (shape[0] != slices.ids[server_id].size()) {
    return Status::ArgumentError("Sparse Partitioner Combiner src shape error");
  }
 
  shape.Set(0, slices.id_size);
  
  Tensor* result;
  if (output->get() == nullptr) {
    output->reset(new WrapperData<Tensor>(type, shape, new initializer::NoneInitializer));
    WrapperData<Tensor>* raw_output = dynamic_cast<WrapperData<Tensor>*>(output->get());
    result = &(raw_output->Internal());
  } else {
    WrapperData<Tensor>* raw_output = dynamic_cast<WrapperData<Tensor>*>(output->get());
    if (raw_output == nullptr) {
      return Status::ArgumentError("Sparse Partitioner Combiner output is not Tensor");
    }
    result = &(raw_output->Internal());
  }

  char* src_ptr = data.Raw<char>();
  char* res_ptr = result->Raw<char>();
  shape.Set(0, 1);
  size_t single_size = shape.NumElements() * SizeOfType(type);
  for (auto item : slices.ids[server_id]) {
    memcpy(res_ptr + item * single_size, src_ptr, single_size);
    src_ptr += single_size;
  }
  return Status::Ok();
}

Status SparseId::Init(PartitionerContext* ctx, Data* src) {
  VariableInfo* info = ctx->GetVariableInfo();
  if (info->type != VariableInfo::kIndex) {
    return Status::ArgumentError("Sparse Partitioner Only Allow by kIndex");
  }
  WrapperData<Tensor>* data_wrapper = dynamic_cast<WrapperData<Tensor>*>(src);
  if (data_wrapper == nullptr) {
    return Status::ArgumentError("Sparse Partitioner Only Allow the Tensor Data");
  }
  Tensor& id = data_wrapper->Internal();

  if (info->parts.size() == 1 && info->shape.empty()) {
    return Status::ArgumentError("Sparse Partitioner Doesn't Accept Scalar Type");
  }
  if (id.Shape().Size() != 1) {
    return Status::ArgumentError("Sparse Parttioner: ID Should be 1-D");
  }

  size_t limit = 0;
  // We need to use lower_bound(splits, id) to find server about id
  std::vector<size_t> splits;
  for (auto item : info->parts) {
    limit += item.size;
    splits.push_back(limit - 1);
  }

  SparseSlices slices;
  slices.ids.resize(info->parts.size());
  slices.id_size = id.Shape()[0];

  CASES(id.Type(), do {
    T* raw_ids = id.Raw<T>();
    for (size_t i = 0; i < id.Shape()[0]; i++) {
      size_t x = raw_ids[i];
      if (x >= limit) {
        return Status::ArgumentError("Sparse Partitioner: id overflow id=" + std::to_string(raw_ids[i]));
      }
      int split_id = std::lower_bound(splits.begin(), splits.end(), x) - splits.begin();
      slices.ids[split_id].push_back(i);
    }
  } while(0));

  ctx->SetData(id_, new WrapperData<SparseSlices>(std::move(slices)));
  return Status::Ok();
}

Status HashId::Init(PartitionerContext* ctx, Data* src) {
  VariableInfo* info = ctx->GetVariableInfo();
  if (info->type != VariableInfo::kHash) {
    return Status::ArgumentError("HashId Partitioner Only Allow by kHash");
  }
  WrapperData<Tensor>* data_wrapper = dynamic_cast<WrapperData<Tensor>*>(src);
  if (data_wrapper == nullptr) {
    return Status::ArgumentError("Sparse Partitioner Only Allow the Tensor Data");
  }
  Tensor& id = data_wrapper->Internal();

  if (id.Shape().Size() != 2 || id.Shape()[1] != 2) {
    return Status::ArgumentError("Sparse Parttioner: ID Should be 2-D, ID Shape should be [?, 2]");
  }

  size_t limit = 0;
  // We need to use lower_bound(splits, id) to find server about id
  std::vector<size_t> splits;
  for (auto item : info->parts) {
    limit += item.size;
    splits.push_back(limit - 1);
  }

  if (limit != Hasher::kTargetRange) {
    return Status::ArgumentError("HashId Parttioner: Variable Info Error, Check the Placementer");
  }

  SparseSlices slices;
  slices.ids.resize(info->parts.size());
  slices.id_size = id.Shape()[0];

  CASES(id.Type(), do {
    T* raw_ids = id.Raw<T>();
    for (size_t i = 0; i < id.Shape()[0]; i++) {
      size_t x = Hasher::Hash128(raw_ids[i * 2], raw_ids[i * 2 + 1]);
      int split_id = std::lower_bound(splits.begin(), splits.end(), x) - splits.begin();
      slices.ids[split_id].push_back(i);
    }
  } while(0));

  ctx->SetData(id_, new WrapperData<SparseSlices>(std::move(slices)));
  return Status::Ok();
}

}
}
}

