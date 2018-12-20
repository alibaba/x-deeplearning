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

#ifndef PS_SERVICE_SEASTAR_LIB_SEASTAR_SERVER_LIB_H_
#define PS_SERVICE_SEASTAR_LIB_SEASTAR_SERVER_LIB_H_

#include <thread>
#include <memory>

#include <core/ps_queue_hub/queue_hub.hh>
#include <service/server_network_context.hh>

#include "server_func_manager.h"

namespace ps {
namespace service {
namespace seastar {

class SeastarServerLib {
 public:
  SeastarServerLib(int port, int core_num);
  bool Start();
  void Stop();
  bool RegisterServerFunc(size_t id, const ServerFunc& server_func);

 private:
  void ToCmdOptions(int* argc, char*** argv);

 private:
  ps::network::ServerNetworkContext server_context_;
  std::shared_ptr<std::thread> server_thread_;
  int port_;
  int core_num_;
  std::string core_ids_;
    
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif //PS_SERVICE_SEASTAR_LIB_SEASTAR_SERVER_LIB_H_
