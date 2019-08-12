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

#ifndef PS_PLUS_CLIENT_PARTITIONER_H_
#define PS_PLUS_CLIENT_PARTITIONER_H_

#include "ps-plus/common/status.h"
#include "ps-plus/common/data.h"
#include "ps-plus/message/variable_info.h"

#include <memory>
#include <vector>

namespace ps {
namespace client {

class PartitionerContext {
 public:
  PartitionerContext();
  PartitionerContext(const VariableInfo& variable_info);
  Data* GetData(size_t id);
  void SetData(size_t id, Data* data);
  void AddDeleter(Data* data);
  void AddDeleter(Data* data, size_t index);
  void SetVariableInfo(const VariableInfo& info);
  VariableInfo* GetVariableInfo();
 private:
  std::vector<std::unique_ptr<Data>> datas_;
  std::vector<std::vector<std::unique_ptr<Data>>> deleter_;
  VariableInfo variable_info_;
};

class Partitioner {
 public:
  virtual ~Partitioner() {}
  virtual Status Init(PartitionerContext* ctx, Data* src);
  virtual Status CombineInit(PartitionerContext* ctx, std::unique_ptr<Data>* output);
  virtual Status Split(PartitionerContext* ctx, Data* src, std::vector<Data*>* dst);
  virtual Status Combine(PartitionerContext* ctx, Data* src, size_t server_id, std::unique_ptr<Data>* output);
};

}
}

#endif
