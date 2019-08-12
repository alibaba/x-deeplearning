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

#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/model_server/model_server.h"
#include "xdl/core/ops/ps_ops/model_server/model_server_forward.h"
#include "xdl/core/ops/ps_ops/model_server/model_server_backward.h"

namespace xdl {

ModelServer::ModelServer(
    const std::string& scheduler_addr, int server_type, int server_id,
    const std::string& forward_cache_spec,
    const std::string& backward_cache_spec)
  : forward_handle_(PsModelServerForwardQueue::Instance()->NewHandle()),
    backward_handle_(PsModelServerBackwardQueue::Instance()->NewHandle()) {
  auto forward = [this] (ps::Tensor ids, ps::modelserver::ForwardCache::Callback cb) {
    PsModelServerForwardQueue::Instance()->Run(forward_handle_, ids, cb);
  };
  auto backward = [this] (ps::Tensor ids, ps::Tensor grads, ps::modelserver::BackwardCache::Callback cb) {
    PsModelServerBackwardQueue::Instance()->Run(backward_handle_, ids, grads, cb);
  };
  ps::modelserver::ModelServer* server = new ps::modelserver::ModelServer(forward, backward, forward_cache_spec, backward_cache_spec);
  ps::modelserver::ModelServerService* service = new ps::modelserver::ModelServerService(server, scheduler_addr, server_type, server_id);
  service_.reset(service);
}
Status ModelServer::Init() {
  return PS2XDL::ConvertStatus(service_->Init());
}

}

