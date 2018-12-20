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

#ifndef PS_SERVICE_SEASTAR_LIB_SEASTAR_SERVER_CLIENT_LIB_H_
#define PS_SERVICE_SEASTAR_LIB_SEASTAR_SERVER_CLIENT_LIB_H_

#include <thread>
#include <memory>

#include <core/ps_queue_hub/queue_hub.hh>
#include <service/server_client_network_context.hh>

#include "server_func_manager.h"

namespace ps {
namespace service {
namespace seastar {

class Closure;

class SeastarServerClientLib {
 public:
  SeastarServerClientLib(int port, 
                         int core_num, 
                         int server_thread_num, 
                         int client_thread_num,
                         bool bind_cores = false,
                         uint64_t timeout = DEFAULT_TIMEOUT);
  bool Start();
  void Stop();
  bool RegisterServerFunc(size_t id, const ServerFunc& server_func);
  void Request(int32_t server_id, 
               int32_t func_id,
               const std::vector<ps::Data*>& request_datas,
               Closure* closure,
               bool delete_request_data = true);

  bool Connect(const int32_t server_id, 
               const std::string& server_addr,
               bool async = false,
               bool connect_in_seastar = false);

 private:
  void ToCmdOptions(int* argc, char*** argv);

 private:
  static const int CONNECT_RETRY_CNT;
  static const int CONNECT_RETRY_SECONDS;
  static const int MAX_SERVER_NUM;
  static const uint64_t DEFAULT_TIMEOUT;

  ps::network::ServerClientNetworkContext context_;
  std::shared_ptr<std::thread> thread_;
  int port_;
  int core_num_;
  int server_thread_num_;
  int client_thread_num_;
  std::string core_ids_;
  bool bind_cores_;
  uint64_t timeout_;
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif //PS_SERVICE_SEASTAR_LIB_SEASTAR_SERVER_LIB_H_
