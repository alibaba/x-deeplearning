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

#ifndef PS_PLUS_CLIENT_CLIENT_H_
#define PS_PLUS_CLIENT_CLIENT_H_

#include <iostream>
#include <memory>

#include "ps-plus/common/logging.h"
#include "ps-plus/client/raw_client.h"
#include "ps-plus/client/base_client.h"
#include "ps-plus/common/tensor.h"


namespace ps {
namespace client {

class Client: public BaseClient {
 public:
  using Callback = std::function<void (const Status&)>;

  Client(RawClient* raw) : raw_(raw) {}

  Status Init() {
    return raw_->Init();
  }

  // Note: element in datas/splitter/combiner will be free after cb run
  void Process(
    const UdfChain& udf, 
    const std::string& var_name,
    const std::vector<Data*>& datas,
    const std::vector<Partitioner*>& splitter,
    const std::vector<Partitioner*>& combiner,
    std::vector<std::unique_ptr<Data> >* results,
    const Callback& cb) override {
    return raw_->Process(udf, var_name, datas, splitter, combiner, results, cb);
  }

  void Process(
    const UdfChain& udf, 
    const std::vector<std::string>& var_names,
    const std::vector<Data*>& datas,
    const std::vector<MergedPartitioner*>& splitter,
    const std::vector<MergedPartitioner*>& combiner,
    std::vector<std::vector<std::unique_ptr<Data> > >* results,
    const Callback& cb) override {
    return raw_->Process(udf, var_names, datas, splitter, combiner, results, cb);
  }

  void Save(const std::string& name, const Callback& cb) override {
    return raw_->Save(name, cb);
  }

  void Restore(const std::string& name, const Callback& cb) override {
    return raw_->Restore(name, cb);
  }

  void TriggerStreamingModelDense(const std::string& stream_ver, const Callback& cb) override {
    return raw_->TriggerStreamingModelDense(stream_ver, cb);
  }

  Status InitGlobalQueue(
      const std::string& name,
      const std::vector<std::string>& paths,
      size_t epochs,
      bool epoch_isolate = false) override {
    return raw_->InitGlobalQueue(name, paths, epochs, epoch_isolate);
  }

  Status GetNextFile(
      const std::string& name,
      size_t worker_id,
      std::string* path,
      size_t* begin,
      size_t* epoch) override {
    return raw_->GetNextFile(name, worker_id, path, begin, epoch);
  }

  Status ReportWorkerState(
      const std::string& name,
      size_t worker_id,
      const std::vector<WorkerState>& worker_states) override {
    return raw_->ReportWorkerState(name, worker_id, worker_states);
  }

  Status RestoreWorkerState(
      const std::string& name,
      size_t worker_id) override {
    return raw_->RestoreWorkerState(name, worker_id);
  }

  void TriggerStreamingModelSparse(const std::string& stream_ver, const Callback& cb) override {
    return raw_->TriggerStreamingModelSparse(stream_ver, cb);
  }

  void TriggerStreamingModelHash(const std::string& stream_ver, const Callback& cb) override {
    return raw_->TriggerStreamingModelHash(stream_ver, cb);
  }

  Status RegisterVariable(const std::string& name, const VariableInfo& info) override {
    return raw_->RegisterVariable(name, info);
  }

  void AsynchronizeEnter(int id, int staleness, int worker_count, const Callback& cb) override {
    raw_->AsynchronizeEnter(id, staleness, worker_count, cb);
  }

  void SynchronizeEnter(int id, int worker_count, const Callback& cb) override {
    sync_mode_ = true;
    worker_count_ = worker_count;
    raw_->SynchronizeEnter(id, worker_count, &token_, cb);
  }

  void SynchronizeLeave(int id, const Callback& cb) override {
    raw_->SynchronizeLeave(id, token_, cb);
  }

  void WorkerReportFinish(int id, const Callback& cb) override {
    raw_->WorkerReportFinish(id, cb);        
  }

  void GetWorkerFinishCount(int64_t* count, const Callback& cb) {
    raw_->GetWorkerFinishCount(count, cb);
  }  

  void WorkerBarrier(int id, int worker_count, const Callback& cb) override {
    raw_->WorkerBarrier(id, worker_count, cb);
  }

  void WorkerBarrierV2(int barrier_id, int task_id, int task_num, int token, const Callback& cb) override {
    raw_->WorkerBarrierV2(barrier_id, task_id, task_num, token, cb);
  }

  Status UpdateVariableVisitInfo(const std::string& name, int64_t id_num) {
    return raw_->UpdateVariableVisitInfo(name, id_num);
  }    
  void ModelServerForward(int type, const Tensor& ids, Tensor* rst, const Callback& cb) override {
    raw_->ModelServerForward(type, ids, rst, cb);
  }
  void ModelServerBackward(int type, const Tensor& ids, const Tensor& grads, const Callback& cb) override {
    raw_->ModelServerBackward(type, ids, grads, cb);
  }

  template <typename... Targs>
  std::vector<Data*> Args(Targs&&... args) {
    return std::vector<Data*>({
      new WrapperData<typename std::remove_cv<typename std::remove_reference<Targs>::type>::type>(std::forward<Targs>(args))...
    });
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
                      const std::vector<float>& save_ratios,
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
                      const std::vector<float>& save_ratios,                      
                      const std::string& updater,
                      const std::vector<Data*>& data,
                      const Callback& cb) override;
  void MergedHashStatis(const std::vector<std::string>& var_names,
                        const std::vector<Tensor>& ids,
                        const std::vector<float>& save_ratios,
                        const std::vector<Tensor>& clicks,
                        const Tensor& global_step,
                        const Tensor& statis_decay,
                        const Tensor& statis_decay_period,
                        const std::string& statis_type,
                        std::vector<Tensor>* result,
                        const Callback& cb) override;

 private:
  Status GetVariableInfo(const std::string& name, VariableInfo* info) {
    return raw_->GetVariableInfo(name, info);
  }

 private:
  std::unique_ptr<RawClient> raw_;
  bool sync_mode_ = false;
  int worker_count_ = -1;
  int64_t token_ = -1;  
};

} //namespace client
} //namespace ps

#endif

