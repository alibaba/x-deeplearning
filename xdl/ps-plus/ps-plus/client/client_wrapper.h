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

#ifndef PS_PLUS_CLIENT_CLIENT_WRAPPER_H_
#define PS_PLUS_CLIENT_CLIENT_WRAPPER_H_

#include "ps-plus/common/data.h"
#include "ps-plus/common/status.h"
#include "ps-plus/client/udf.h"
#include "ps-plus/client/partitioner.h"
#include "ps-plus/client/merged_partitioner.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/message/worker_state.h"
#include <vector>
#include <functional>

namespace ps {
namespace client {

class ClientWrapper {
 public:
  using Callback = std::function<void (const Status&)>;

  virtual ~ClientWrapper() {}

  // Change Internal cluster version.
  virtual Status ConnectToCluster(const std::string& addr) = 0;

  // Add Internal cluster version to request.
  virtual void UpdateVariableInfo(const std::vector<VariableInfo>& input, std::vector<VariableInfo>* output, const Callback& cb) = 0;
  virtual void UpdateVariableVisitInfo(const std::string& name, int64_t id_num, const Callback& cb) = 0;
  virtual void Process(const std::string& var_name, size_t server_id, size_t udf_id, const std::vector<Data*>& input, std::vector<Data*>* output, const Callback& cb) = 0;
  virtual void RegisterUdf(size_t server_id, const UdfChain& def, const Callback& cb) = 0;
  virtual void Save(const std::string& version, const Callback& cb) = 0;
  virtual void Restore(const std::string& version, const Callback& cb) = 0;
  virtual Status InitGlobalQueue(const std::string& name, const std::vector<std::string>& paths, size_t epochs, bool epoch_isolate = false) = 0;
  virtual Status GetNextFile(const std::string& name, size_t worker_id, std::string* path, size_t* begin, size_t* epoch) = 0;
  virtual Status ReportWorkerState(const std::string& name, size_t worker_id, const std::vector<WorkerState>& worker_states) = 0;
  virtual Status RestoreWorkerState(const std::string& name, size_t worker_id) = 0;
  virtual void ModelServerForward(int server_type, int server_id, const Tensor& ids, std::unique_ptr<Tensor>* rst, const Callback& cb) = 0;
  virtual void ModelServerBackward(int server_type, int server_id, const Tensor& ids, const Tensor& grads, const Callback& cb) = 0;
  virtual void TriggerStreamingModelDense(const std::string& stream_ver, const Callback& cb) = 0;
  virtual void TriggerStreamingModelSparse(const std::string& stream_ver, const Callback& cb) = 0;
  virtual void TriggerStreamingModelHash(const std::string& stream_ver, const Callback& cb) = 0;
  virtual void AsynchronizeEnter(int id, int staleness, int worker_count, const Callback& cb) = 0;
  virtual void SynchronizeEnter(int id, int worker_count, int64_t* token, const Callback& cb) = 0;    
  virtual void SynchronizeLeave(int id, int64_t token, const Callback& cb) = 0;
  virtual void WorkerReportFinish(int id, const Callback& cb) = 0;
  virtual void GetWorkerFinishCount(int64_t* count, const Callback& cb) = 0;
  virtual void WorkerBarrier(int id, int worker_count, const Callback& cb) = 0;    
  virtual void WorkerBarrierV2(int barrier_id, int task_id, int task_num, int token, const Callback& cb) = 0;
  virtual int ServerSize(int id) = 0;
  virtual int ServerTypeSize() = 0;
};

}
}

#endif
