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

#ifndef PS_PLUS_CLIENT_PARTITIONER_MERGED_HASH_H_
#define PS_PLUS_CLIENT_PARTITIONER_MERGED_HASH_H_

#include "ps-plus/client/merged_partitioner.h"
#include "ps-plus/client/partitioner/sparse.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/tensor_shape.h"

namespace ps {
namespace client {
namespace partitioner {

class MergedHashData : public MergedPartitioner {
 public:
  MergedHashData(size_t id = 0) : id_(id) {}
  virtual Status Split(MergedPartitionerContext* ctx, Data* src, std::vector<Data*>* dst) override;
  virtual Status Combine(MergedPartitionerContext* ctx, Data* src, size_t server_id, std::vector<std::unique_ptr<Data>>* output) override;
  virtual Status CombineInit(MergedPartitionerContext* ctx, std::vector<std::unique_ptr<Data>>* output) override;
 protected:
  size_t id_;
  HashId id_partitioner_;
  HashData data_partitioner_;
};

class MergedHashId : public MergedHashData {
 public:
  MergedHashId(size_t id = 0) : MergedHashData(id) {}
  virtual Status Init(MergedPartitionerContext* ctx, Data* src) override;
};

}
}
}

#endif

