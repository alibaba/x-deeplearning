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

#ifndef PS_SERVICE_SEASTAR_LIB_SEASTAR_CLIENT_LIB_H_
#define PS_SERVICE_SEASTAR_LIB_SEASTAR_CLIENT_LIB_H_

#include <thread>
#include <memory>
#include <tuple>
#include <vector>
#include <service/client_network_context.hh>

namespace ps {
class Data;
}

namespace ps {
namespace service {
namespace seastar {

class Closure;
class SeastarRequestSerializer;

class SeastarClientLib {
 public:
  using ServerAddr = std::tuple<int64_t, std::string>;
  SeastarClientLib(const std::vector<ServerAddr>& server_addrs,
                   int user_thread_num, 
                   int core_num,
                   uint64_t timeout = DEFAULT_TIMEOUT);

  bool Start();
  void Stop();
  void Request(int32_t server_id, 
               int32_t func_id,
               const std::vector<ps::Data*>& request_datas,
               Closure* closure,
               bool delete_request_data = true);

  bool Connect(const int32_t server_id, 
               const std::string& server_addr);

  int CheckServer();
 private:
  void ToCmdOptions(int* argc, char*** argv);

 private:
  static const int CONNECT_RETRY_CNT;
  static const int CONNECT_RETRY_SECONDS;
  static const int MAX_SERVER_NUM;
  static const uint64_t DEFAULT_TIMEOUT;

  std::vector<ServerAddr> server_addrs_;
  int user_thread_num_;
  int core_num_;
  uint64_t timeout_;
  std::string core_ids_;
  ps::network::ClientNetworkContext client_context_;
  std::shared_ptr<std::thread> client_thread_;
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif // PS_SERVICE_SEASTAR_LIB_SEASTAR_CLIENT_LIB_H_
