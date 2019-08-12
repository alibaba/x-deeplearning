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

#ifndef PS_PLUS_CLIENT_PARTITIONER_SPARSE_H_
#define PS_PLUS_CLIENT_PARTITIONER_SPARSE_H_

#include "ps-plus/client/partitioner.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/tensor_shape.h"
#include "ps-plus/common/qrw_lock.h"

namespace ps {
namespace client {
namespace partitioner {

class SparseData : public Partitioner {
 public:
  SparseData(size_t id = 0) : id_(id) {}
  virtual Status Split(PartitionerContext* ctx, Data* src, std::vector<Data*>* dst) override;
  virtual Status Combine(PartitionerContext* ctx, Data* src, size_t server_id, std::unique_ptr<Data>* output) override;
  virtual Status CombineInit(PartitionerContext* ctx, std::unique_ptr<Data>* output) override;
 protected:
  size_t id_;
  QRWLock lock_;
};

class SparseId : public SparseData {
 public:
  SparseId(size_t id = 0) : SparseData(id) {}
  virtual Status Init(PartitionerContext* ctx, Data* src) override;
};

using HashData = SparseData;

class HashId : public HashData {
 public:
  HashId(size_t id = 0) : HashData(id) {}
  virtual Status Init(PartitionerContext* ctx, Data* src) override;
};

}
}
}

#endif

