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

#ifndef PS_MODEL_SERVER_MODEL_SERVER_SERVICE_H_
#define PS_MODEL_SERVER_MODEL_SERVER_SERVICE_H_

#include "ps-plus/common/status.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/thread_pool.h"
#include "ps-plus/message/version.h"
#include "ps-plus/message/func_ids.h"
#include "ps-plus/service/seastar/lib/seastar_server_client_lib.h"
#include "ps-plus/service/seastar/lib/done_closure.h"
#include "ps-plus/model_server/model_server.h"

#include <vector>
#include <string>

namespace ps {
namespace modelserver {

class ModelServerService {
 public:
  ModelServerService(
      ModelServer* model_server, const std::string& scheduler,
      int server_type, int server_id);

  ~ModelServerService();
  
  Status Init();

  void Flush(
      const std::vector<Data*>& inputs,
      std::vector<Data*>* outputs,
      ps::service::seastar::DoneClosure* done);
  void RequestForward(
      const std::vector<Data*>& inputs,
      std::vector<Data*>* outputs,
      ps::service::seastar::DoneClosure* done);
  void RequestBackward(
      const std::vector<Data*>& inputs,
      std::vector<Data*>* outputs,
      ps::service::seastar::DoneClosure* done);

  void RegisterServer();

 private:
  static const int CLIENT_THREAD_NUM = 100;

  int server_type_;
  int server_id_;
  int port_;
  std::string ip_;
  int core_num_;
  bool stop_;
  Version server_version_;
  std::string scheduler_kv_addr_;

  std::unique_ptr<ModelServer> model_server_;
  std::unique_ptr<std::thread> register_server_loop_;
  std::unique_ptr<ps::service::seastar::SeastarServerClientLib> seastar_lib_;
  std::atomic<Version> ver_;
};

}
}

#endif
