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

#ifndef PS_PLUS_CLIENT_MODEL_SERVER_SPLITTER_H_
#define PS_PLUS_CLIENT_MODEL_SERVER_SPLITTER_H_

#include "ps-plus/common/status.h"
#include "ps-plus/common/data.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/message/variable_info.h"

#include <memory>
#include <vector>

namespace ps {
namespace client {

class ModelServerSplitter {
 public:
  Status Init(int server_size, Tensor ids);
  Status Split(Tensor t, std::vector<Tensor>* rst);
  Status Combine(int id, Tensor data, Tensor* rst);
 private:
  std::vector<size_t> ids_;
  std::vector<std::vector<size_t>> server_ids_;
  int server_size_;
};

}
}

#endif
