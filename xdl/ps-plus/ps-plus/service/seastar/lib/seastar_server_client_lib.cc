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

#include "seastar_server_client_lib.h"

#include <thread>
#include <chrono>
#include <exception>
#include <service/session_context.hh>
#include <service/connect_item.hh>
#include <service/stop_item.hh>
#include <service/seastar_exception.hh>
#include <core/ps_queue_hub/queue_hub.hh>
#include <core/ps_queue_hub/queue_work_item.hh>

#include "ps-plus/common/data.h"
#include "ps-plus/common/net_utils.h"
#include "common.h"
#include "cpu_pool.h"
#include "closure_manager.h"
#include "seastar_request_item.h"
#include "server_func_manager.h"

using namespace ps::network;

namespace ps {
namespace service {
namespace seastar {

const int SeastarServerClientLib::CONNECT_RETRY_CNT = 12;
const int SeastarServerClientLib::CONNECT_RETRY_SECONDS = 5;
const int SeastarServerClientLib::MAX_SERVER_NUM = 5000;
const uint64_t SeastarServerClientLib::DEFAULT_TIMEOUT = 30 * 60 * 1000;

SeastarServerClientLib::SeastarServerClientLib(int port,
        int core_num,
        int server_thread_num,
        int client_thread_num,
        bool bind_cores,
        uint64_t timeout)
  : context_(MAX_SERVER_NUM, 
             1, core_num, 
             client_thread_num, 
             server_thread_num)
  , port_(port)
  , core_num_(core_num)
  , server_thread_num_(server_thread_num)
  , client_thread_num_(client_thread_num)
  , bind_cores_(bind_cores)
  , timeout_(timeout) {
}

bool SeastarServerClientLib::Start() {
  if (!CPUPool::GetInstance()->Allocate(core_num_, &core_ids_)) {
    std::cerr << "allocate core[" << core_num_ << "] failed!" << std::endl;
    return false;
  }
  
  int argc;
  char** argv;
  ToCmdOptions(&argc, &argv);
  thread_.reset(new std::thread([this, argc, argv] {
    context_(argc, argv, timeout_);
  }));

  sleep(2);
  return true;
}

void SeastarServerClientLib::ToCmdOptions(int* argc, char*** argv) {
  *argc = 8;
  *argv = new char*[9];
  (*argv)[0] = new char[1000];
  (*argv)[1] = new char[1000];
  (*argv)[2] = new char[1000];
  (*argv)[3] = new char[1000];
  (*argv)[4] = new char[1000];
  (*argv)[5] = new char[1000];
  (*argv)[6] = new char[1000];
  (*argv)[7] = new char[1000];
  (*argv)[8] = nullptr;
  snprintf((*argv)[0], 1000, "--smp=%d", core_num_);
  snprintf((*argv)[1], 1000, "--cpuset=%s", core_ids_.c_str());
  snprintf((*argv)[2], 1000, "--port=%d", port_);
  strcpy((*argv)[3], "--tcp_nodelay_on=1");
  strcpy((*argv)[4], "--tcp_keep_alive_idle=300");
  strcpy((*argv)[5], "--tcp_keep_alive_cnt=6");
  strcpy((*argv)[6], "--tcp_keep_alive_interval=10");
  snprintf((*argv)[7], 1000, "--thread-affinity=%d", bind_cores_ ? 1 : 0);
  printf("seastar binding cores: %s\n", bind_cores_ ? "true" : "false");
}

void SeastarServerClientLib::Stop() {
  int thread_id = QueueHub<Item>::GetPortNumberOnThread();
  ps::network::StopItem* item = new ps::network::StopItem(thread_id);
  context_.GetClient().Stop(item, thread_id, 0, NULL);  
  thread_->join();
}

bool SeastarServerClientLib::RegisterServerFunc(size_t id, 
                                                const ServerFunc& server_func) {
  return ServerFuncManager::GetInstance()->RegisterServerFunc(
      id, server_func) == 0;
}

void SeastarServerClientLib::Request(int32_t server_id, 
                                     int32_t func_id,
                                     const std::vector<ps::Data*>& request_datas,
                                     Closure* closure,
                                     bool delete_request_data) {
  int thread_id = QueueHub<Item>::GetPortNumberOnThread();
  SeastarRequestItem* request_item = 
    new SeastarRequestItem(server_id,
                           thread_id,
                           func_id, 
                           &(context_.GetClient()), 
                           request_datas,
                           delete_request_data);
  size_t core_id = context_.GetClient().GetCoreIdOfServerId(server_id);
  context_.GetClient().EnqueueItem(request_item, thread_id, core_id, closure, false);
}

bool SeastarServerClientLib::Connect(int32_t server_id, 
                                     const std::string& addr,
                                     bool async,
                                     bool connect_in_seastar) {
  size_t core_id = context_.GetClient().GetCoreIdOfServerId(server_id);
  int thread_id = QueueHub<Item>::GetPortNumberOnThread();
  for (size_t i = 0; i < CONNECT_RETRY_CNT; ++i) {
    if (!connect_in_seastar) {
      ConnectRequestItem* item = 
        new ConnectRequestItem(&(context_.GetClient()), 
                               thread_id,
                               core_id, 
                               server_id, 
                               addr,
                               false);
      auto f0 = context_.GetClient().ConnectToServer(item, thread_id, core_id, NULL);
      sleep(5);
      context_.GetClient().SetResponseProcessorIdOfServer(
          server_id, SEASTAR_RESPONSE_PROCESSOR_ID);
      if (async) return true;
      try {
        f0.Get();
        return true;
      } catch (ps::network::ConnectException& e) {
        std::this_thread::sleep_for(std::chrono::seconds(CONNECT_RETRY_SECONDS));
        continue;
      }
    } else {
      context_.GetClient().mShardClients.invoke_on(core_id, 
                                                   &SeastarClient::ConnectToOne,
                                                   server_id,
                                                   addr,
                                                   false).then([this, server_id](auto code) {
                                                     context_.GetClient().SetResponseProcessorIdOfServer(
                                                         server_id, SEASTAR_RESPONSE_PROCESSOR_ID);
                                                     return ::seastar::make_ready_future<int>(0);
                                                   });
      return true;
    }
  }

  return false;
}

} // namespace seastar
} // namespace service
} // namespace ps
