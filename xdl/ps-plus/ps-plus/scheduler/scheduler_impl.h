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

#ifndef PS_SCHEDULER_SCHEDULER_IMPL_H_
#define PS_SCHEDULER_SCHEDULER_IMPL_H_

#include <condition_variable>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "ps-plus/common/status.h"
#include "ps-plus/message/cluster_info.h"
#include "ps-plus/message/variable_info.h"
#include "ps-plus/message/version.h"
#include "ps-plus/common/thread_pool.h"
#include "ps-plus/message/streaming_model_manager.h"

#include "placementer.h"
#include "scheduler_service.h"
#include "synchronizer.h"

namespace ps {
namespace scheduler {

enum OpCode {
  kNone, kSave, kRestore
};

using OpCallback = std::function<void (const ps::Status&)>;

class SchedulerImpl {
 public:
  SchedulerImpl(
      const std::string& server_count,
      const std::string& scheduler_addr,
      const std::string& checkpoint_path,
      const Placementer::Arg& placement_arg,
      const std::string& streaming_dense_model_addr,
      const std::string& streaming_sparse_model_addr,
      const std::string& streaming_hash_model_addr,
      bool bind_cores = false);
  ~SchedulerImpl();

  Status Start();

  Version GetVersion();
  Status RegisterServer(const ServerInfo& server);
  Status GetClusterInfo(const Version version, ClusterInfo* result);
  void Save(Version version, const std::string& checkpoint, OpCallback cb);
  void Restore(Version version, const std::string& checkpoint, OpCallback cb);
  void TriggerStreamingDense(Version version, OpCallback cb);
  void TriggerStreamingSparse(Version version, OpCallback cb);
  void TriggerStreamingHash(Version version, OpCallback cb);
  ps::Status UpdateVariableInfo(Version version,
                                const std::vector<VariableInfo>& info,
                                std::vector<VariableInfo>* result);
  void AsynchronizeEnter(Version version, int id, int staleness, int worker_count, 
                        std::function<void (const Status&)> cb);
  void SynchronizeEnter(Version version, int id, int worker_count, 
                        std::function<void (int64_t, const Status&)> cb);
  void SynchronizeLeave(Version version, int id, int64_t token, std::function<void (const Status&)> cb);
  void WorkerReportFinish(Version version, int id, std::function<void (const Status&)> cb);    
  void WorkerBarrier(Version version, int id, int worker_count, std::function<void (const Status&)> cb);
  ps::Status UpdateVariableVisitInfo(Version version, const std::string& var_name, int64_t ids);
  ps::Status WriteMetaInfo();

 private:
  std::unique_ptr<std::thread> main_thread_;
  std::unique_ptr<std::thread> meta_thread_;
  bool stopped_;

  std::mutex m_;
  bool ready_;
  Version version_;
  std::map<int32_t, std::function<void (const Status&)>> worker_barriers_;
  int worker_count_ = 0;
  std::set<int32_t> finished_workers_;
  std::map<std::pair<ServerType, ServerId>, ServerInfo> servers_;

  ps::Status VersionMismatch(Version exp, Version act);

  OpCode op_code_;
  std::string op_checkpoint_;
  OpCallback op_cb_;
  std::condition_variable op_cv_;
  std::string OpName(OpCode code);
  void WaitForOp();
  void AssignOp(OpCode code, Version version, const std::string& checkpoint,
                OpCallback cb);

  void Main();
  void MainLoop();

  void WaitForServers();

  std::vector<ps::VariableInfo> variable_info_;
  Status InternalRestore(const std::string& checkpoint);
  Status InternalSave(const std::string& checkpoint);
  Status InternalTriggerStreamingDense(Version version);
  Status InternalTriggerStreamingSparse(Version version);
  Status InternalTriggerStreamingHash(Version version);
  void InternalAsynchronizeEnter(Version version, int id,
                                int staleness, int worker_count,
                                std::function<void (const Status&)> cb);
  void InternalSynchronizeEnter(Version version, int id, int worker_count,
                                std::function<void (int64_t, const Status&)> cb);
  void InternalSynchronizeLeave(Version version, int id, int64_t token,
                                std::function<void (const Status&)> cb);
  void InternalWorkerReportFinish(Version version, int id, std::function<void (const Status&)> cb);
  void InternalWorkerBarrier(Version version, int id, int worker_count, std::function<void (const Status&)> cb);    
    
  Placementer* placementer_;
  const std::string checkpoint_path_;
  const Placementer::Arg placement_arg_;
  std::string vp_string_;
  std::string meta_string_;

  std::unique_ptr<SchedulerService> service_;
  std::unique_ptr<ThreadPool> lazy_queue_;
  std::unique_ptr<ThreadPool> synchronizer_queue_;

  std::unique_ptr<SyncMechanism> sync_;

  std::string streaming_dense_model_addr_;
  std::string streaming_sparse_model_addr_;
  std::string streaming_hash_model_addr_;
  std::unique_ptr<StreamingModelWriter> streaming_dense_model_writer_;
};

}
}

#endif // PS_SCHEDULER_SCHEDULER_IMPL_H_
