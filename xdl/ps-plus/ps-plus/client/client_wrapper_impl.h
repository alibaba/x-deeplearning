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

#ifndef PS_PLUS_CLIENT_CLIENT_SERVICE_H_
#define PS_PLUS_CLIENT_CLIENT_SERVICE_H_

#include "ps-plus/client/client_wrapper.h"
#include "ps-plus/service/seastar/lib/seastar_client_lib.h"
#include "ps-plus/service/seastar/lib/event_client_lib.h"
#include "ps-plus/service/seastar/lib/callback_closure.h"
#include "ps-plus/message/version.h"
#include "ps-plus/message/cluster_info.h"
#include "ps-plus/message/func_ids.h"

namespace ps {
namespace client {

class ClientWrapperImpl : public ClientWrapper {
 public:
  ClientWrapperImpl() {}
  Status ConnectToCluster(const std::string& addr) override;
  void UpdateVariableInfo(const std::vector<VariableInfo>& input, std::vector<VariableInfo>* output, const Callback& cb) override;
  void UpdateVariableVisitInfo(const std::string& name, int64_t id_num, const Callback& cb) override;
  void Process(const std::string& var_name, size_t server_id, size_t udf_id, const std::vector<Data*>& input, std::vector<Data*>* output, const Callback& cb) override;
  void RegisterUdf(size_t server_id, const UdfChain& def, const Callback& cb) override;
  void Save(const std::string& version, const Callback& cb) override;
  void Restore(const std::string& version, const Callback& cb) override;
  Status InitGlobalQueue(const std::string& name, const std::vector<std::string>& paths, size_t epochs, bool epoch_isolate = false) override;
  Status GetNextFile(const std::string& name, size_t worker_id, std::string* path, size_t* begin, size_t* epoch) override;
  Status ReportWorkerState(const std::string& name, size_t worker_id, const std::vector<WorkerState>& worker_states) override;
  Status RestoreWorkerState(const std::string& name, size_t worker_id) override;
  void ModelServerForward(int server_type, int server_id, const Tensor& ids, std::unique_ptr<Tensor>* rst, const Callback& cb) override;
  void ModelServerBackward(int server_type, int server_id, const Tensor& ids, const Tensor& grads, const Callback& cb) override;
  void TriggerStreamingModelDense(const std::string& stream_ver, const Callback& cb) override;
  void TriggerStreamingModelSparse(const std::string& stream_ver, const Callback& cb) override;
  void TriggerStreamingModelHash(const std::string& stream_ver, const Callback& cb) override;
  void AsynchronizeEnter(int id, int staleness, int worker_count, const Callback& cb) override;
  void SynchronizeEnter(int id, int worker_count, int64_t* token, const Callback& cb) override;    
  void SynchronizeLeave(int id, int64_t token, const Callback& cb) override;
  void WorkerReportFinish(int id, const Callback& cb) override;
  void GetWorkerFinishCount(int64_t* count, const Callback& cb);
  void WorkerBarrier(int id, int worker_count, const Callback& cb) override;    
  void WorkerBarrierV2(int barrier_id, int task_id, int task_num, int token, const Callback& cb) override;
  int ServerSize(int id) override;
  int ServerTypeSize() override;

  //using ClientLib = ps::service::seastar::SeastarClientLib;
  using ClientLib = ps::service::seastar::EventClientLib;
 private:
  Status CreateServerLib();
  Status ConnectToScheduler(const std::string& addr);
  Status WaitForReady();
  Status ConnectToServers();

  ClientLib* client_lib_;
  Version scheduler_version_;

  static ClientLib* client_lib_singleton_;
  std::vector<size_t> offset_;
  std::mutex mu_;
};

} //namespace client
} //namespace ps

#endif
