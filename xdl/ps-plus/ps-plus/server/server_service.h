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

#ifndef PS_PLUS_SERVER_SERVER_SERVICE_H_
#define PS_PLUS_SERVER_SERVER_SERVICE_H_

#include "ps-plus/message/version.h"
#include "ps-plus/message/func_ids.h"
#include "ps-plus/server/server.h"
#include "ps-plus/common/thread_pool.h"
#include "ps-plus/service/seastar/lib/seastar_server_client_lib.h"
#include "ps-plus/service/seastar/lib/done_closure.h"

#include <atomic>

namespace ps {
namespace server {

class ServerService {
 public:
  ServerService(const std::string& scheduler, 
		int server_id,
		std::string streaming_dense_model_addr,
		std::string streaming_sparse_model_addr,
    std::string streaming_hash_model_addr,
    bool bind_cores);
  Status Init();
  ~ServerService();
 private:
  void RegisterUdfChain(const std::vector<Data*>& inputs, std::vector<Data*>* outputs);
  void Process(const std::vector<Data*>& inputs, std::vector<Data*>* outputs);
  void Save(const std::vector<Data*>& inputs, std::vector<Data*>* outputs);
  void Restore(const std::vector<Data*>& inputs, std::vector<Data*>* outputs);
  void Announce(const std::vector<Data*>& inputs, std::vector<Data*>* outputs);
  void StreamingDenseVarName(const std::vector<Data*>& inputs, std::vector<Data*>* outputs);
  void GatherStreamingDenseVar(const std::vector<Data*>& inputs, std::vector<Data*>* outputs);
  void TriggerStreamingSparse(const std::vector<Data*>& inputs, std::vector<Data*>* outputs);
  void TriggerStreamingHash(const std::vector<Data*>& inputs, std::vector<Data*>* outputs);
  void RegisterServer();

  static const int CLIENT_THREAD_NUM = 100;

  std::unique_ptr<std::thread> register_server_loop_;
  int port_;
  std::string ip_;
  bool stop_;
  int server_id_;
  int core_num_;
  int bind_cores_;
  
  Version server_version_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<ps::service::seastar::SeastarServerClientLib> seastar_lib_;
  std::string scheduler_kv_addr_;
  std::atomic<Version> scheduler_version_;
  std::unique_ptr<ThreadPool> lazy_queue_;
  std::string streaming_dense_model_addr_;
  std::string streaming_sparse_model_addr_;
  std::string streaming_hash_model_addr_;
};

}
}

#endif

