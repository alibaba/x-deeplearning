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
    std::vector<std::unique_ptr<Data>>* results,
    const Callback& cb) override {
    return raw_->Process(udf, var_name, datas, splitter, combiner, results, cb);
  }

  void Save(const std::string& name, const Callback& cb) override {
    return raw_->Save(name, cb);
  }

  void Restore(const std::string& name, const Callback& cb) override {
    return raw_->Restore(name, cb);
  }

  void TriggerStreamingModelDense(const Callback& cb) override {
    return raw_->TriggerStreamingModelDense(cb);
  }

  void TriggerStreamingModelSparse(const Callback& cb) override {
    return raw_->TriggerStreamingModelSparse(cb);
  }

  void TriggerStreamingModelHash(const Callback& cb) override {
    return raw_->TriggerStreamingModelHash(cb);
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

  void WorkerBarrier(int id, int worker_count, const Callback& cb) override {
    raw_->WorkerBarrier(id, worker_count, cb);
  }    

  Status UpdateVariableVisitInfo(const std::string& name, int64_t id_num) {
    return raw_->UpdateVariableVisitInfo(name, id_num);
  }

  Status UpdateVariableShowInfo(const std::string& name, const Tensor& ids) {
    return raw_->UpdateVariableShowInfo(name, ids);
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
                double add_probability, 
                Tensor* result, 
                const Callback& cb) override;
  void HashPush(const std::string& variable_name, 
                const Tensor& ids, 
                const std::string& updater, 
                const std::vector<Data*>& data, 
                const Callback& cb) override;

 private:
  std::unique_ptr<RawClient> raw_;
  bool sync_mode_ = false;
  int worker_count_ = -1;
  int64_t token_ = -1;  
};

} //namespace client
} //namespace ps

#endif


