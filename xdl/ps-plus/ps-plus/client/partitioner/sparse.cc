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

Status SplitOneHashId(PartitionerContext* ctx, const Tensor& id, size_t index) {
  VariableInfo* info = ctx->GetVariableInfo();
  if (info->type == VariableInfo::kHash128) {
    if (id.Shape().Size() != 2 || id.Shape()[1] != 2) {
      return Status::ArgumentError("HashId Parttioner: Hash128 ID Should be 2-D, ID Shape should be [?, 2], variable[" + info->name + "]");
    }
  } else {
    if (id.Shape().Size() != 1) {
      return Status::ArgumentError("HashId Parttioner: Hash64 ID Should be 1-D, variable[" + info->name + "]");
    }
  }

  size_t limit = 0;
  // We need to use lower_bound(splits, id) to find server about id
  std::vector<size_t> splits;
  for (auto& item : info->parts) {
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
      size_t x;
      if (info->type == VariableInfo::kHash128) {
        x = Hasher::Hash128(raw_ids[i * 2], raw_ids[i * 2 + 1]);
      } else {
        x = Hasher::Hash64(raw_ids[i]);
      }
      int split_id = std::lower_bound(splits.begin(), splits.end(), x) - splits.begin();
      slices.ids[split_id].push_back(i);
    }
  } while(0));

  ctx->SetData(index, new WrapperData<SparseSlices>(std::move(slices)));
  return Status::Ok();
}

Status SplitOneSparseData(PartitionerContext* ctx, const Tensor& data, std::vector<Data*>* dst, size_t id) {
  if (data.Shape().IsScalar()) {
    return Status::ArgumentError("Sparse Partitioner Doesn't Accept Scalar Type");
  }

  WrapperData<SparseSlices>* id_wrapper = dynamic_cast<WrapperData<SparseSlices>*>(ctx->GetData(id));
  if (id_wrapper == nullptr) {
    return Status::ArgumentError("Sparse Partitioner Init Error");
  }
  SparseSlices& slices = id_wrapper->Internal();

  if (data.Shape()[0] != slices.id_size) {
    return Status::ArgumentError("Data's Shape Does Not Match the ID Size");
  }

  dst->resize(slices.ids.size());
  
  TensorShape shape = data.Shape();
  DataType type = data.Type();
  shape.Set(0, 1);
  size_t single_size = shape.NumElements() * SizeOfType(type);
  MultiThreadDoTBB(slices.ids.size(), [&](const Range& r) {
        for (size_t i = r.begin; i < r.end; i++) {
          auto& item = slices.ids[i];
          TensorShape shape = data.Shape();
          shape.Set(0, item.size());
          WrapperData<Tensor>* result = new WrapperData<Tensor>(data.Type(), shape, new initializer::NoneInitializer);
          char* res_ptr = result->Internal().Raw<char>();
          char* src_ptr = data.Raw<char>();
          for (auto id : item) {
            memcpy(res_ptr, src_ptr + id * single_size, single_size);
            res_ptr += single_size;
          }
          ctx->AddDeleter(result, i);
          (*dst)[i] = result;
        }
        return Status::Ok();
      });
  return Status::Ok();
}

}

Status SparseData::Split(PartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
  if (dynamic_cast<WrapperData<Tensor>*>(src) != nullptr) {
    Tensor& data = dynamic_cast<WrapperData<Tensor>*>(src)->Internal();
    return SplitOneSparseData(ctx, data, dst, id_);
  } else if (dynamic_cast<WrapperData<std::vector<Tensor>>*>(src) != nullptr) {
    VariableInfo* info = ctx->GetVariableInfo();
    std::vector<Tensor>& data_vec = dynamic_cast<WrapperData<std::vector<Tensor>>*>(src)->Internal();
    dst->clear();
    for (size_t i = 0; i < info->parts.size(); ++i) {
      WrapperData<std::vector<Tensor>>* result = new WrapperData<std::vector<Tensor>>();
      dst->emplace_back(result);
      ctx->AddDeleter(result);
    }
    for (size_t i = 0; i < data_vec.size(); ++i) {
      std::vector<Data*> one_dst;
      Status one_status = SplitOneSparseData(ctx, data_vec[i], &one_dst, i);
      if (!one_status.IsOk()) {
        return one_status;
      }
      for(size_t j = 0; j < one_dst.size(); ++j) {
        dynamic_cast<WrapperData<std::vector<Tensor>>*>((*dst)[j])->Internal().push_back(dynamic_cast<WrapperData<Tensor>*>(one_dst[j])->Internal());
      }
    }
    return Status::Ok();
  } else {
    return Status::ArgumentError("SparseData Partitioner Only Allow the Tensor Data or Tensor Data Vector");
  }
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
 
  Tensor* result;
  shape.Set(0, slices.id_size);

  /*
  {
    QRWLocker lock(lock_, QRWLocker::kWrite);
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
  }
  */
  WrapperData<Tensor>* raw_output = dynamic_cast<WrapperData<Tensor>*>(output->get());
  if (raw_output == nullptr) {
    return Status::ArgumentError("Sparse Partitioner Combiner output is not Tensor");
  }
  result = &(raw_output->Internal());

  char* res_ptr = result->Raw<char>();
  shape.Set(0, 1);
  size_t single_size = shape.NumElements() * SizeOfType(type);
  MultiThreadDo(slices.ids[server_id].size(), [&](const Range& r) {
        char* src_ptr = data.Raw<char>() + r.begin * single_size;
        for (size_t i = r.begin; i < r.end; ++i) {
          size_t item = slices.ids[server_id][i];
          memcpy(res_ptr + item * single_size, src_ptr, single_size);
          src_ptr += single_size;
        }
        return Status::Ok();
      }, 1000);
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
  if (info->type != VariableInfo::kHash128 && info->type != VariableInfo::kHash64) {
    return Status::ArgumentError("HashId Partitioner Only Allow by kHash");
  }
  if (dynamic_cast<WrapperData<Tensor>*>(src) != nullptr) {
    Tensor& id = dynamic_cast<WrapperData<Tensor>*>(src)->Internal();
    return SplitOneHashId(ctx, id, id_);
  } else if (dynamic_cast<WrapperData<std::vector<Tensor>>*>(src) != nullptr) {
    std::vector<Tensor>& id_vec = dynamic_cast<WrapperData<std::vector<Tensor>>*>(src)->Internal();
    for (size_t i = 0; i < id_vec.size(); ++i) {
      Status one_status = SplitOneHashId(ctx, id_vec[i], i);
      if (!one_status.IsOk()) {
        return one_status;
      }
    }
    return Status::Ok();
  } else {
    return Status::ArgumentError("HashId Partitioner Only Allow the Tensor Data or Tensor Data Vector");
  }
}

Status SparseData::CombineInit(PartitionerContext* ctx, std::unique_ptr<Data>* output) {
  WrapperData<SparseSlices>* id_wrapper = dynamic_cast<WrapperData<SparseSlices>*>(ctx->GetData(id_));
  if (id_wrapper == nullptr) {
    return Status::ArgumentError("Sparse Partitioner Init Error");
  }
  SparseSlices& slices = id_wrapper->Internal();

  VariableInfo* info = ctx->GetVariableInfo();

  std::vector<size_t> dims;
  dims.push_back(slices.id_size);
  for (size_t i = 1; i < info->shape.size(); ++i) {
    dims.push_back(info->shape[i]);
  }
  TensorShape shape(dims);
  DataType type = info->datatype;
  output->reset(new WrapperData<Tensor>(type, shape, new initializer::NoneInitializer));

  return Status::Ok();
}

}
}
}

