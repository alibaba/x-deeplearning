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

#ifndef PS_PLUS_SERVER_STREAMING_MODEL_ARGS_H_
#define PS_PLUS_SERVER_STREAMING_MODEL_ARGS_H_

#include "ps-plus/message/streaming_model_manager.h"

namespace ps {
namespace server {

struct StreamingModelArgs {
  std::string streaming_dense_model_addr;
  std::string streaming_sparse_model_addr;
  std::string streaming_hash_model_addr;
  std::unique_ptr<StreamingModelWriter> streaming_sparse_model_writer;
  std::unique_ptr<StreamingModelWriter> streaming_hash_model_writer;
  StreamingModelArgs() = default;
  StreamingModelArgs(const StreamingModelArgs& rhs)
    : streaming_dense_model_addr(rhs.streaming_dense_model_addr),
      streaming_sparse_model_addr(rhs.streaming_sparse_model_addr),
      streaming_hash_model_addr(rhs.streaming_hash_model_addr) {}
  Status Init() {
    if (!streaming_sparse_model_addr.empty()) {
      PS_CHECK_STATUS(StreamingModelManager::OpenWriterAny(streaming_sparse_model_addr, &streaming_sparse_model_writer));
    }
    if (!streaming_hash_model_addr.empty()) {
      PS_CHECK_STATUS(StreamingModelManager::OpenWriterAny(streaming_hash_model_addr, &streaming_hash_model_writer));
    }
    return Status::Ok();
  }
};

}
}

#endif

