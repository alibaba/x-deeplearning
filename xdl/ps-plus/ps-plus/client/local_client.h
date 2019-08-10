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

#ifndef PS_PLUS_CLIENT_LOCAL_CLIENT_H_
#define PS_PLUS_CLIENT_LOCAL_CLIENT_H_

#include <memory>
#include <unordered_map>
#include <mutex>

#include "ps-plus/server/local_server.h"
#include "ps-plus/client/base_client.h"
#include "ps-plus/common/global_file_queue.h"

namespace ps {
namespace client {

// LocalClient is only used in inference

class LocalClient: public BaseClient {
 public:
  using Callback = std::function<void (const Status&)>;
  LocalClient(const std::string& checkpoint_path) {
    ckpt_path_ = checkpoint_path;
  }

  Status Init() override {
    local_server_.reset(new ps::server::LocalServer(ckpt_path_));
    return local_server_->Init();
  }

  void Process(const UdfChain& udf, 
	       const std::string& var_name,
	       const std::vector<Data*>& datas,
	       const std::vector<Partitioner*>& splitter,
	       const std::vector<Partitioner*>& combiner,
	       std::vector<std::unique_ptr<Data> >* results,
	       const Callback& cb) override {
    return Process(udf, var_name, datas, results, cb);
  }
  
  void Process(const UdfChain& udf, 
           const std::vector<std::string>& var_names,
           const std::vector<Data*>& datas,
           const std::vector<MergedPartitioner*>& splitter,
           const std::vector<MergedPartitioner*>& combiner,
           std::vector<std::vector<std::unique_ptr<Data> > >* results,
           const Callback& cb) override {
    results->emplace_back();
    Process(udf, "^hash_variable", datas, &((*results)[0]), cb);
  }

  void Process(const UdfChain& udf, 
               const std::string& var_name,
               const std::vector<Data*>& datas,
               std::vector<std::unique_ptr<Data> >* results,
               const Callback& cb);
               
  void Save(const std::string& name, const Callback& cb) override {
    Status st = local_server_->Save(name);
    cb(st);
  }

  void Restore(const std::string& ckpt, const Callback& cb) override {
    Status st = local_server_->Restore(ckpt);
    cb(st);
  }

  Status InitGlobalQueue(
      const std::string& name,
      const std::vector<std::string>& paths,
      size_t epochs,
      bool epoch_isolate = false) override {
    std::unique_lock<std::mutex> lock(mu_);
    auto it = global_file_queues_.find(name);
    if (it == global_file_queues_.end()) {
      global_file_queues_[name].reset(new GlobalFileQueue());
    }

    return global_file_queues_[name]->Init(
        paths, epochs, epoch_isolate);
  }

  Status GetNextFile(
      const std::string& name,
      size_t worker_id,
      std::string* path,
      size_t* begin,
      size_t* epoch) override {
    std::unique_lock<std::mutex> lock(mu_);
    WorkerState file;
    Status st = global_file_queues_[name]->GetNextFile(
        worker_id, & file);
    *path = file.path_;
    *begin = file.begin_;
    *epoch = file.epoch_;
    return st;
  }

  Status ReportWorkerState(
      const std::string& name,
      size_t worker_id,
      const std::vector<WorkerState>& worker_states) override {
    std::unique_lock<std::mutex> lock(mu_);
    return global_file_queues_[name]->ReportWorkerState(
        worker_id, worker_states);
  }

  Status RestoreWorkerState(
      const std::string& name,
      size_t worker_id) override {
    std::unique_lock<std::mutex> lock(mu_);
    return global_file_queues_[name]->RestoreWorkerState(worker_id);
  }
               
  Status RegisterVariable(const std::string& name, 
                          const VariableInfo& info) override {
    return local_server_->RegisterVariable(name, info);
  }

  void IndexInitializer(const std::string& variable_name, 
                        Initializer* init, 
                        const Callback& cb) override;

  void IdentityInitializer(const std::string& variable_name, 
                           const Tensor& init, 
                           const Callback& cb) override;

  void HashInitializer(const std::string& variable_name, 
                       Initializer* init, 
                       const Callback& cb) override;

  void IsInitialized(const std::string& variable_name, 
                     bool* inited, 
                     const Callback& cb) override;

  void DensePull(const std::string& variable_name, 
                 Tensor* result, 
                 const Callback& cb) override;

  void DensePush(const std::string& variable_name, 
                 const std::string& updater, 
                 const std::vector<Data*>& data, 
                 const Callback& cb) override;

  void SparsePull(const std::string& variable_name, 
                  const Tensor& ids, 
                  Tensor* result, 
                  const Callback& cb) override;

  void SparsePush(const std::string& variable_name, 
                  const Tensor& ids, 
                  const std::string& updater, 
                  const std::vector<Data*>& data, 
                  const Callback& cb) override;

  void HashPull(const std::string& variable_name, 
                const Tensor& ids,
                const float& save_ratio,
                Tensor* result, 
                const Callback& cb) override;

  void MergedHashPull(const std::vector<std::string>& var_names, 
                      const std::vector<Tensor>& ids,
                      const std::vector<float>& save_ratio,
                      std::vector<Tensor>* result, 
                      const Callback& cb) override;

  void HashPush(const std::string& variable_name, 
                const Tensor& ids,
                const float& save_ratio,
                const bool& insertable,
                const std::string& updater,
                const std::vector<Data*>& data, 
                const Callback& cb) override;

  void MergedHashPush(const std::vector<std::string>& var_names,
                      const std::vector<Tensor>& ids,
                      const std::vector<float>& save_ratio,
                      const std::string& updater,
                      const std::vector<Data*>& data,
                      const Callback& cb) override;

  void MergedHashStatis(const std::vector<std::string>& var_names,
                        const std::vector<Tensor>& ids,
                        const std::vector<float>& save_ratio,
                        const std::vector<Tensor>& clicks,
                        const Tensor& global_step,
                        const Tensor& statis_decay,
                        const Tensor& statis_decay_period,
                        const std::string& statis_type,
                        std::vector<Tensor>* result,
                        const Callback& cb) override;

  void ModelServerForward(int type, const Tensor& ids, Tensor* rst, const Callback& cb) override {
    cb(Status::ArgumentError("Not Supported in Local"));
  }
  void ModelServerBackward(int type, const Tensor& ids, const Tensor& grads, const Callback& cb) override {
    cb(Status::ArgumentError("Not Supported in Local"));
  }

  /* not used in inference and local train */
  void TriggerStreamingModelDense(const std::string& stream_ver, const Callback& cb) override {
    cb(Status::Ok());
  }

  void TriggerStreamingModelSparse(const std::string& stream_ver, const Callback& cb) override {
    cb(Status::Ok());
  }

  void TriggerStreamingModelHash(const std::string& stream_ver, const Callback& cb) override {
    cb(Status::Ok());
  }
  void AsynchronizeEnter(int id, 
                         int staleness, 
                         int worker_count, 
                         const Callback& cb) override {
    cb(Status::Ok());
  }

  void SynchronizeEnter(int id, 
                        int worker_count, 
                        const Callback& cb) override {
    cb(Status::Ok());
  }

  void SynchronizeLeave(int id, const Callback& cb) override {
    cb(Status::Ok());
  }

  void WorkerReportFinish(int id, const Callback& cb) override {
    cb(Status::Ok());
  }

  void GetWorkerFinishCount(int64_t* id, const Callback& cb) override {
    cb(Status::Ok());
  }
               
  void WorkerBarrier(int id, int worker_count, const Callback& cb) override {
    cb(Status::Ok());
  }    

  void WorkerBarrierV2(int barrier_id, int task_id, int task_num, int token, const Callback& cb) override {
    cb(Status::Ok());
  }

 private:
  Status GetVariableInfo(const std::string& name, VariableInfo* info) {
    return local_server_->GetVariableInfo(name, info);
  }

 private:
  // a local server holds all variables in memory
  std::unique_ptr<ps::server::LocalServer> local_server_;
  std::string ckpt_path_;
  std::mutex mu_;
  std::unordered_map<std::string, std::unique_ptr<GlobalFileQueue> > global_file_queues_;
};

} //namespace client
} //namespace ps

#endif

