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

#include "ps-plus/model_server/backward.h"

namespace ps {
namespace modelserver {

class BackwardNoCache : public BackwardCache {
 public:
  Status Init(BackwardRun backward, const std::unordered_map<std::string, std::string>& map) override {
    backward_ = backward;
    return Status::Ok();
  }
  void Calc(Tensor ids, Tensor grads, Callback cb) override {
    backward_(ids.Clone(), grads.Clone(), cb);
  }
  Status Flush() override {
    return Status::Ok();
  }
 private:
  BackwardRun backward_;
};

BACKWARD_REGISTER(BackwardNoCache, no_cache);

}
}

