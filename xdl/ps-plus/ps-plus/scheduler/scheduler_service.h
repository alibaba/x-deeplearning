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

#ifndef PS_SCHEDULER_SCHEDULER_SERVICE_H_
#define PS_SCHEDULER_SCHEDULER_SERVICE_H_

#include "ps-plus/common/data.h"
#include "ps-plus/common/status.h"
#include "ps-plus/common/net_utils.h"
#include "ps-plus/service/seastar/lib/seastar_server_client_lib.h"
#include "ps-plus/service/seastar/lib/done_closure.h"
#include "ps-plus/message/func_ids.h"
#include "ps-plus/message/version.h"
#include "ps-plus/message/variable_info.h"
#include "ps-plus/message/streaming_model_manager.h"
#include "ps-plus/message/streaming_model_infos.h"

namespace ps {
namespace scheduler {

class SchedulerImpl;
class SchedulerService {
 public:
  SchedulerService(SchedulerImpl* impl, 
                   const std::string& server_count, 
                   const std::string& scheduler_kv_addr,
                   bool bind_cores)
    : impl_(impl)
    , core_num_(NetUtils::GetAvailableCpuNum())
    , scheduler_kv_addr_(scheduler_kv_addr) 
    , port_(NetUtils::GetAvailablePort())
    , bind_cores_(bind_cores) {
    server_count_ = server_count;
    server_offset_.push_back(0);
    int s = 0;
    int y = 0;
    for (char c : server_count) {
      if ('0' <= c && c <= '9') {
        y = y * 10 + (c - '0');
      } else {
        s += y;
        server_offset_.push_back(s);
        y = 0;
      }
    }
    server_offset_.push_back(s + y);
  }
  ~SchedulerService();
  Status Start();
  void SetServer(int server_type, int server_id, const std::string& server_addr);
  void ServerSave(
      int server_type,
      int server_id,
      Version version,
      const std::string& checkpoint,
      const std::vector<VariableInfo>& info,
      std::function<void(Status)> cb);
  void ServerRestore(
      int server_type,
      int server_id,
      Version version,
      const std::vector<VariableInfo>& from,
      const std::vector<VariableInfo>& to,
      std::function<void(Status)> cb);
  void ServerStreamingDenseVarName(
      int server_type,
      int server_id,
      Version version,
      std::function<void(Status, const DenseVarNames& vars)> cb);
  void ServerGatherStreamingDenseVar(
      int server_type,
      int server_id,
      Version version,
      const DenseVarNames& vars,
      std::function<void(Status, const DenseVarValues& vars)> cb);
  void ServerTriggerStreamingSparse(
      int server_type,
      int server_id,
      Version version,
      const std::string& stream_version,
      std::function<void(Status)> cb);
  void ServerTriggerStreamingHash(
      int server_type,
      int server_id,
      Version version,
      const std::string& stream_version,
      std::function<void(Status)> cb);
  void ModelServerFlush(
      int server_type,
      int server_id,
      Version version,
      std::function<void(Status)> cb);
  int GetServerSize(int server_type);
  int GetServerTotalSize();
  int GetServerTypeSize();
 private:
  void GetVersion(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void RegisterServer(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void GetClusterInfo(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void Save(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void Restore(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void InitGlobalQueue(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void GetNextFile(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void ReportWorkerState(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void RestoreWorkerState(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void UpdateVariableInfo(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void TriggerStreamingDense(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void TriggerStreamingSparse(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void TriggerStreamingHash(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void SynchronizeEnter(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void SynchronizeLeave(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void AsynchronizeEnter(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void WorkerReportFinish(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void GetWorkerFinishCount(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);
  void WorkerBarrier(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);    
  void WorkerBarrierV2(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);    
  void UpdateVariableVisitInfo(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done);

  static const int CLIENT_THREAD_NUM = 100;

  SchedulerImpl* impl_;
  std::string server_count_;
  bool bind_cores_;
  int core_num_;
  std::unique_ptr<ps::service::seastar::SeastarServerClientLib> seastar_lib_;

  std::string scheduler_kv_addr_;
  int port_;
  std::string ip_;
  int server_type_size_;
  std::vector<int> server_offset_;
};

}
}

#endif // PS_SCHEDULER_SCHEDULER_SERVICE_H_

