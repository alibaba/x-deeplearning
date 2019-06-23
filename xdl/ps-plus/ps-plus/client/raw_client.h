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

#ifndef PS_PLUS_CLIENT_RAW_CLIENT_H_
#define PS_PLUS_CLIENT_RAW_CLIENT_H_

#include "ps-plus/client/client_wrapper.h"
#include "ps-plus/client/udf.h"
#include "ps-plus/client/udf.h"
#include "ps-plus/common/status.h"
#include "ps-plus/common/data.h"

#include <functional>
#include <string>
#include <mutex>
#include <future>
#include <unordered_set>

namespace ps {
namespace client {

struct ClientArgs {
  std::string scheduler_addr;
  std::function<ClientWrapper*()> client_wrapper_creator;
  std::unordered_map<std::string, VariableInfo> variable_info;
};

class RawClient {
 public:
  using Callback = std::function<void (const Status&)>;

  RawClient(const ClientArgs& args);

  Status Init();

  // Note: element in datas/splitter/combiner will be free after cb run
  void Process(
    const UdfChain& udf, 
    const std::string& var_name,
    const std::vector<Data*>& datas,
    const std::vector<Partitioner*>& splitter,
    const std::vector<Partitioner*>& combiner,
    std::vector<std::unique_ptr<Data>>* results,
    const Callback& cb);

  void ModelServerForward(int type, const Tensor& ids, Tensor* rst, const Callback& cb);
  void ModelServerBackward(int type, const Tensor& ids, const Tensor& grads, const Callback& cb);

  Status RegisterVariable(const std::string& name, const VariableInfo& info);

  void Save(const std::string& name, const Callback& cb);
  void Restore(const std::string& name, const Callback& cb);
  void TriggerStreamingModelDense(const Callback& cb);
  void TriggerStreamingModelSparse(const Callback& cb);
  void TriggerStreamingModelHash(const Callback& cb);

  void AsynchronizeEnter(int id, int staleness, int worker_count, const Callback& cb);
  void SynchronizeEnter(int id, int worker_count, int64_t* token, const Callback& cb);    
  void SynchronizeLeave(int id, int64_t token, const Callback& cb);
  void WorkerReportFinish(int id, const Callback& cb);    
  void WorkerBarrier(int id, int worker_count, const Callback& cb);
  Status UpdateVariableVisitInfo(const std::string& name, int64_t id_num);

 private:
  Status GetVariableInfo(const std::string& name, VariableInfo* info);
  void Process(const std::string& var_name, size_t server_id, const UdfChain& udf, const std::vector<Data*>& input, std::vector<Data*>* output, const Callback& cb);

  ClientArgs args_;
  std::unique_ptr<ClientWrapper> client_wrapper_;
  std::mutex variable_info_mutex_;
  std::unordered_map<std::string, VariableInfo> variable_infos_;
  bool init_variable_info_;
};

} //namespace client
} //namespace ps

#endif

