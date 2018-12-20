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

#include "ps-plus/model_server/model_server.h"

namespace ps {
namespace modelserver {

ModelServer::ModelServer(
    ForwardCache::ForwardRun forward_factory,
    BackwardCache::BackwardRun backward_factory,
    const std::string& forward_spec, const std::string& backward_spec) {
  forward_factory_ = forward_factory;
  backward_factory_ = backward_factory;
  forward_spec_ = forward_spec;
  backward_spec_ = backward_spec;
}

Status ModelServer::Init() {
  PS_CHECK_STATUS(ForwardCache::Get(
      forward_factory_, forward_spec_, &forward_cache_));
  PS_CHECK_STATUS(BackwardCache::Get(
      backward_factory_, backward_spec_, &backward_cache_));
  return Status::Ok();
}

void ModelServer::Flush(std::function<void(Status)> cb) {
  cb(FlushImpl());
}

void ModelServer::RequestForward(
    Tensor ids, std::function<void(Status, Tensor)> cb) {
  forward_cache_->Calc(ids, cb);
}

void ModelServer::RequestBackward(
    Tensor ids, Tensor grads, std::function<void(Status)> cb) {
  backward_cache_->Calc(ids, grads, cb);
}

Status ModelServer::FlushImpl() {
  PS_CHECK_STATUS(forward_cache_->Flush());
  PS_CHECK_STATUS(backward_cache_->Flush());
  return Status::Ok();
}

}
}
