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

#include "seastar_client_lib.h"

#include <thread>
#include <chrono>
#include <limits>
#include <exception>
#include <service/session_context.hh>
#include <service/connect_item.hh>
#include <service/stop_item.hh>

#include <service/seastar_exception.hh>

#include "ps-plus/common/data.h"
#include "common.h"
#include "cpu_pool.h"
#include "closure_manager.h"
#include "seastar_request_item.h"

#include <core/ps_queue_hub/queue_hub.hh>
#include <core/ps_queue_hub/queue_work_item.hh>

using namespace ps::network;

namespace ps {
namespace service {
namespace seastar {

const int SeastarClientLib::CONNECT_RETRY_CNT = 1;
const int SeastarClientLib::CONNECT_RETRY_SECONDS = 5;
const int SeastarClientLib::MAX_SERVER_NUM = 1000;
const uint64_t SeastarClientLib::DEFAULT_TIMEOUT = 30 * 60 * 1000;

SeastarClientLib::SeastarClientLib(const std::vector<ServerAddr>& server_addrs,
                                   int user_thread_num,
                                   int core_num,
				   uint64_t timeout) 
  : server_addrs_(server_addrs)
  , user_thread_num_(user_thread_num)
  , core_num_(core_num)
  , timeout_(timeout)
  , client_context_(MAX_SERVER_NUM, 1, core_num, user_thread_num) {
}

bool SeastarClientLib::Start() {
  if (!CPUPool::GetInstance()->Allocate(core_num_, &core_ids_)) {
    std::cerr << "allocate core[" << core_num_ << "] failed!" << std::endl;
    return false;
  }

  int argc;
  char** argv;
  ToCmdOptions(&argc, &argv);
  client_thread_.reset(new std::thread([this, argc, argv] {
    client_context_(argc, argv, server_addrs_, timeout_);
  }));
  
  sleep(2);
  return true;
}

void SeastarClientLib::ToCmdOptions(int* argc, char*** argv) {
  *argc = 7;
  *argv = new char*[8];
  (*argv)[0] = new char[1000];
  (*argv)[1] = new char[1000];
  (*argv)[2] = new char[1000];
  (*argv)[3] = new char[1000];
  (*argv)[4] = new char[1000];
  (*argv)[5] = new char[1000];
  (*argv)[6] = new char[1000];
  (*argv)[7] = nullptr;
  snprintf((*argv)[0], 1000, "--smp=%d", core_num_);
  snprintf((*argv)[1], 1000, "--cpuset=%s", core_ids_.c_str());
  strcpy((*argv)[2], "--tcp_nodelay_on=1");
  strcpy((*argv)[3], "--tcp_keep_alive_idle=300");
  strcpy((*argv)[4], "--tcp_keep_alive_cnt=6");
  strcpy((*argv)[5], "--tcp_keep_alive_interval=10");
  strcpy((*argv)[6], "--thread-affinity=0");
}

void SeastarClientLib::Stop() {
  int thread_id = QueueHub<Item>::GetPortNumberOnThread();
  ps::network::StopItem* item = new ps::network::StopItem(thread_id);
  client_context_.Stop(item, thread_id, 0, NULL);  
  client_thread_->join();
}

void SeastarClientLib::Request(int32_t server_id, 
                               int32_t func_id,
                               const std::vector<ps::Data*>& request_datas,
                               Closure* closure,
                               bool delete_request_data) {
  int thread_id = QueueHub<Item>::GetPortNumberOnThread();
  SeastarRequestItem* request_item = 
    new SeastarRequestItem(server_id,
                           thread_id,
                           func_id, 
                           &client_context_, 
                           request_datas,
                           delete_request_data);
  size_t core_id = client_context_.GetCoreIdOfServerId(server_id);
  client_context_.EnqueueItem(request_item, thread_id, core_id, closure, false);
  // std::cout << "put callback[" << cb_id << "]" << 
  //   " server_id:" << server_id <<
  //   " thread_id:" << thread_id <<
  //   " core_id:" << core_id <<
  //   " func_id:" << func_id  << 
  //   " seq:" << request_item->GetSequence() << std::endl;
}

bool SeastarClientLib::Connect(int32_t server_id, 
                               const std::string& addr) {
  size_t core_id = client_context_.GetCoreIdOfServerId(server_id);
  int32_t thread_id = QueueHub<Item>::GetPortNumberOnThread();
  for (size_t i = 0; i < CONNECT_RETRY_CNT; ++i) {
    ConnectRequestItem* item = 
      new ConnectRequestItem(&client_context_, 
                             thread_id,
                             core_id, 
                             server_id, 
                             addr,
                             false);
    auto f0 = client_context_.ConnectToServer(item, thread_id, core_id, NULL);
    try {
      f0.Get();
      client_context_.SetResponseProcessorIdOfServer(
          server_id, SEASTAR_RESPONSE_PROCESSOR_ID);
      return true;
    } catch (ps::network::ConnectException& e) {
      std::this_thread::sleep_for(std::chrono::seconds(CONNECT_RETRY_SECONDS));
      continue;
    }
  }

  return false;
}

int SeastarClientLib::CheckServer() {
  return 0;
}

} // namespace seastar
} // namespace service
} // namespace ps
