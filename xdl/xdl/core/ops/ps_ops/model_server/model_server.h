/* Copyright 2018 Alibaba Group. All Rights Reserved.

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

#ifndef XDL_CORE_OPS_PS_OPS_MODEL_SERVER_MODEL_SERVER_H_
#define XDL_CORE_OPS_PS_OPS_MODEL_SERVER_MODEL_SERVER_H_

#include "xdl/core/lib/status.h"
#include "xdl/core/lib/singleton.h"
#include "xdl/core/lib/concurrent_queue.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "ps-plus/model_server/model_server_service.h"

namespace xdl {

class ModelServer {
 public:
  ModelServer(const std::string& scheduler_addr, int server_type, int server_id,
              const std::string& forward_cache_spec,
              const std::string& backward_cache_spec);
  Status Init();
  int ForwardHandle() { return forward_handle_; }
  int BackwardHandle() { return backward_handle_; }
 private:
  int forward_handle_, backward_handle_;
  std::unique_ptr<ps::modelserver::ModelServerService> service_;
};

}  //namespace xdl

#endif  // XDL_CORE_OPS_PS_OPS_MODEL_SERVER_MODEL_SERVER_H_

