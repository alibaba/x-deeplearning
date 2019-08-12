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

#ifndef PS_MESSAGE_STREAMING_MODEL_MANAGER_H_
#define PS_MESSAGE_STREAMING_MODEL_MANAGER_H_

#include "ps-plus/common/tensor.h"
#include "ps-plus/common/status.h"
#include "ps-plus/common/plugin.h"

namespace ps {

class StreamingModelWriter {
 public:
  struct DenseModel {
    std::string name;
    Tensor data;
  };
  struct SparseModel {
    std::string name;
    Tensor data;
    std::vector<int64_t> ids;
    std::vector<size_t> offsets;
  };
  struct HashModel {
    std::string name;
    Tensor data;
    std::vector<std::pair<int64_t, int64_t>> ids;
    std::vector<size_t> offsets;
    std::vector<std::pair<int64_t, int64_t>> del_ids;
  };
  virtual ~StreamingModelWriter() {}
  virtual Status WriteDenseModel(const std::vector<DenseModel>& val, const std::string& stream_version) = 0;
  virtual Status WriteSparseModel(const std::vector<SparseModel>& val, const std::string& stream_version, const int& server_id) = 0;
  virtual Status WriteHashModel(const std::vector<HashModel>& val, const std::string& stream_version, const int& server_id) = 0;
};

class StreamingModelManager {
 public:
  virtual ~StreamingModelManager() {}
  virtual Status OpenWriter(const std::string& path, std::unique_ptr<StreamingModelWriter>* writer) = 0;
  static Status GetManager(const std::string& path, StreamingModelManager** manager);
  static Status OpenWriterAny(const std::string& path, std::unique_ptr<StreamingModelWriter>* writer);
};

}

#endif // PS_MESSAGE_STREAMING_MODEL_MANAGER_H_

