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

#include "xdl/core/backend/mxnet/mxnet_runner.h"
#include "xdl/core/backend/mxnet/convert_utils.h"
#include <memory>
#include <string>

namespace xdl {

struct MxnetRunnerHolder {
  std::unique_ptr<MxnetRunner> mxnet_runner_;
  std::string graph_def_;
  std::string device_type_;
  std::vector<std::string> var_names_;
  std::unique_ptr<mxnet::cpp::Context> context_;
  bool is_training_;
  bool has_init_grad_;
  int64_t gradient_size_;
};

class MxnetRunnerHolderManager : Singleton<MxnetRunnerHolderManager> {
 public:
  static Status
  GetRunner(int64_t id, MxnetRunnerHolder** holder, std::function<Status(MxnetRunnerHolder&)> init) {
    return Get()->GetRunnerImpl(id, holder, init);
  }
 private:
  Status
  GetRunnerImpl(int64_t id, MxnetRunnerHolder** holder, std::function<Status(MxnetRunnerHolder&)> init) {
    std::unique_lock<std::mutex> lock(mu_);
    if (holders_[id] == nullptr) {
      holders_[id].reset(new MxnetRunnerHolder);
      XDL_CHECK_STATUS(init(*holders_[id]));
    }
    *holder = holders_[id].get();
    return Status::Ok();
  }
  std::mutex mu_;
  std::unordered_map<int64_t, std::unique_ptr<MxnetRunnerHolder>> holders_;
};

}
