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

#ifndef PS_MODEL_SERVER_MODEL_SERVER_H_
#define PS_MODEL_SERVER_MODEL_SERVER_H_

#include "ps-plus/common/status.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/model_server/forward.h"
#include "ps-plus/model_server/backward.h"

namespace ps {
namespace modelserver {

class ModelServer {
 public:
  ModelServer(
      ForwardCache::ForwardRun forward_factory,
      BackwardCache::BackwardRun backward_factory,
      const std::string& forward_spec, const std::string& backward_spec);
  
  Status Init();

  void Flush(std::function<void(Status)> cb);
  void RequestForward(Tensor ids, std::function<void(Status, Tensor)> cb);
  void RequestBackward(Tensor ids, Tensor grads, std::function<void(Status)> cb);

 private:
  Status FlushImpl();

  ForwardCache::ForwardRun forward_factory_;
  BackwardCache::BackwardRun backward_factory_;
  std::string forward_spec_;
  std::string backward_spec_;
  std::unique_ptr<ForwardCache> forward_cache_;
  std::unique_ptr<BackwardCache> backward_cache_;
};

}
}

#endif
