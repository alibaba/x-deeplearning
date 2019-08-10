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

#ifndef PS_PLUS_CLIENT_MERGED_PARTITIONER_H_
#define PS_PLUS_CLIENT_MERGED_PARTITIONER_H_

#include "ps-plus/common/status.h"
#include "ps-plus/common/data.h"
#include "ps-plus/message/variable_info.h"
#include "ps-plus/client/partitioner.h"

#include <memory>
#include <vector>

namespace ps {
namespace client {

class MergedPartitionerContext {
 public:
  MergedPartitionerContext();
  Data* GetData(std::size_t id);
  void SetData(std::size_t id, Data* data);
  void AddDeleter(Data* data);
  void SetContext(std::size_t id, PartitionerContext* context);
  PartitionerContext* GetContext(std::size_t id);
  void AddContext(PartitionerContext* context);
  size_t ContextSize();
 private:
  std::vector<std::unique_ptr<Data>> datas_;
  std::vector<std::unique_ptr<Data>> deleter_;
  std::vector<std::unique_ptr<PartitionerContext>> context_;
};

class MergedPartitioner {
 public:
  virtual ~MergedPartitioner() {}
  virtual Status Init(MergedPartitionerContext* ctx, Data* src);
  virtual Status CombineInit(MergedPartitionerContext* ctx, std::vector<std::unique_ptr<Data>>* output);
  virtual Status Split(MergedPartitionerContext* ctx, Data* src, std::vector<Data*>* dst);
  virtual Status Combine(MergedPartitionerContext* ctx, Data* src, size_t server_id, std::vector<std::unique_ptr<Data>>* output);
};

}
}

#define CHECK_COUNTER(COUNTER, OK) do { if (--COUNTER == 0) {OK.set_value(true);} return;} while(0);

#endif
