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

#ifndef PS_PLUS_CLIENT_BASE_CLIENT_H_
#define PS_PLUS_CLIENT_BASE_CLIENT_H_

#include <memory>

#include "ps-plus/common/data.h"
#include "ps-plus/common/status.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/message/variable_info.h"

#include "ps-plus/client/udf.h"
#include "ps-plus/client/partitioner.h"

namespace ps {
namespace client {

class BaseClient {
 public:
  using Callback = std::function<void (const Status&)>;
  BaseClient() {}
  virtual ~BaseClient() {}
  virtual Status Init() = 0;
  virtual void Save(const std::string& name, const Callback& cb) = 0;
  virtual void Restore(const std::string& name, const Callback& cb) = 0;
  virtual void TriggerStreamingModelDense(const Callback& cb) = 0;
  virtual void TriggerStreamingModelSparse(const Callback& cb) = 0;
  virtual void TriggerStreamingModelHash(const Callback& cb) = 0;

  virtual Status RegisterVariable(const std::string& name, const VariableInfo& info) = 0;

  virtual void AsynchronizeEnter(int id, int staleness, int worker_count, const Callback& cb) = 0; 
  virtual void SynchronizeEnter(int id, int worker_count, const Callback& cb) = 0;
  virtual void SynchronizeLeave(int id, const Callback& cb) = 0;
  virtual void WorkerReportFinish(int id, const Callback& cb) = 0;
  virtual void WorkerBarrier(int id, int worker_count, const Callback& cb) = 0;
  virtual void ModelServerForward(int type, const Tensor& ids, Tensor* rst, const Callback& cb) = 0;
  virtual void ModelServerBackward(int type, const Tensor& ids, const Tensor& grads, const Callback& cb) = 0;

  virtual void IndexInitializer(const std::string& variable_name, 
                                Initializer* init, 
                                const Callback& cb) = 0;
  virtual void IdentityInitializer(const std::string& variable_name, 
                                   const Tensor& init, 
                                   const Callback& cb) = 0;
  virtual void HashInitializer(const std::string& variable_name, 
                               Initializer* init, 
                               const Callback& cb) = 0;
  virtual void IsInitialized(const std::string& variable_name, 
                             bool* inited, 
                             const Callback& cb) = 0;
  virtual void DensePull(const std::string& variable_name, 
                         Tensor* result, 
                         const Callback& cb) = 0;
  virtual void DensePush(const std::string& variable_name, 
                         const std::string& updater, 
                         const std::vector<Data*>& data, 
                         const Callback& cb) = 0;
  virtual void SparsePull(const std::string& variable_name, 
                          const Tensor& ids, 
                          Tensor* result, 
                          const Callback& cb) = 0;
  virtual void SparsePush(const std::string& variable_name, 
                          const Tensor& ids, 
                          const std::string& updater, 
                          const std::vector<Data*>& data, 
                          const Callback& cb) = 0;
  virtual void HashPull(const std::string& variable_name, 
                        const Tensor& ids, 
                        double add_probability,
                        Tensor* result, 
                        const Callback& cb) = 0;
  virtual void HashPush(const std::string& variable_name, 
                        const Tensor& ids, 
                        const std::string& updater, 
                        const std::vector<Data*>& data, 
                        const Callback& cb) = 0;

  virtual void Process(const UdfChain& udf, 
		       const std::string& var_name,
		       const std::vector<Data*>& datas,
		       const std::vector<Partitioner*>& splitter,
		       const std::vector<Partitioner*>& combiner,
		       std::vector<std::unique_ptr<Data>>* results,
		       const Callback& cb) = 0;

  template <typename... Targs>
  std::vector<Data*> Args(Targs&&... args) {
    return std::vector<Data*>({
      new WrapperData<typename std::remove_cv<typename std::remove_reference<Targs>::type>::type>(std::forward<Targs>(args))...
    });
  }
};

} //namespace client
} //namespace ps

#endif

