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

#include "xdl/core/ops/ps_ops/client.h"

#include <memory>
#include <iostream>

#include <ps-plus/client/client_wrapper_impl.h>
#include "ps-plus/client/local_client.h"
#include "ps-plus/client/client.h"

namespace xdl {

namespace {
std::unique_ptr<ps::client::BaseClient> current_client;
}

bool ConnectToClient(const std::string& addr, const std::string& ckpt_path) {
  if (current_client != nullptr) {
    return false;
  }

  if (addr == "localhost") {
    current_client.reset(new ps::client::LocalClient(ckpt_path));  
  } else {
    ps::client::ClientArgs args;
    args.scheduler_addr = addr;
    args.client_wrapper_creator = [](){return new ps::client::ClientWrapperImpl();};
    ps::client::RawClient* raw_client = new ps::client::RawClient(args);
    current_client.reset(new ps::client::Client(raw_client));
  }

  ps::Status st = current_client->Init();
  if (!st.IsOk()) {
    current_client.reset(nullptr);
    return false;
  }

  return true;
}

bool RestartClient() {
  if (current_client == nullptr) {
    return false;
  }

  ps::Status st = current_client->Init();
  if (!st.IsOk()) {
    return false;
  }

  return true;
}

bool ResetClient() {
  current_client = nullptr;
  return true;
}

bool Connected() {
  return current_client != nullptr;
}

Status GetClient(ps::client::BaseClient** result) {
  if (current_client == nullptr) {
    return Status::Internal("Client is not initialized");
  }
  *result = current_client.get();
  return Status::Ok();
}

}

using xdl::ConnectToClient;
using xdl::RestartClient;
using xdl::Connected;

extern "C" {
  extern bool TFPS_CONNECT_TO_CLIENT(const char* addr, const char* ckpt_path) {
    return ConnectToClient(addr, ckpt_path);
  }
  extern bool TFPS_RESTART_CLIENT() {
    return RestartClient();
  }
  extern bool TFPS_CONNECTED() {
    return Connected();
  }
}
