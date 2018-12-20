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

#include "ps-plus/client/local_client.h"
#include "ps-plus/client/partitioner/dense.h"
#include "ps-plus/client/partitioner/logic.h"
#include "ps-plus/client/partitioner/sparse.h"
#include "ps-plus/client/partitioner/broadcast.h"
#include "ps-plus/client/partitioner/index.h"
#include "ps-plus/client/partitioner/hash.h"

#include <iostream>

namespace ps {
namespace client {

#define RETURN_ASYNC(STATUS) do { cb(STATUS); return; } while (0)

#define CHECK_ASYNC(STATUS) do {                                                                                \
    Status st_ = STATUS;                                                                                        \
    if (!st_.IsOk()) {                                                                                          \
        st_.Msg() += "\nCHECKED BY [" #STATUS "] @ FILE[" __FILE__ "] LINE[" + std::to_string(__LINE__) + "]";  \
        RETURN_ASYNC(st_);                                                                                      \
    }                                                                                                           \
} while (0)

void LocalClient::IndexInitializer(const std::string& variable_name, 
                                   Initializer* init, 
                                   const LocalClient::Callback& cb) {
  VariableInfo info;
  CHECK_ASYNC(local_server_->GetVariableInfo(variable_name, &info));
  std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  std::vector<Data*> inputs = Args(
      info.datatype, TensorShape(dims), (size_t)0, 
      std::unique_ptr<Initializer>(init));
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("IndexVariableInitializer", 
              UdfData(0), 
              UdfData(1), 
              UdfData(2), 
              UdfData(3));
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, '^' + variable_name, inputs, outputs, realcb);
}

void LocalClient::IdentityInitializer(const std::string& variable_name, 
                                      const Tensor& init, 
                                      const LocalClient::Callback& cb) {
  VariableInfo info;
  CHECK_ASYNC(local_server_->GetVariableInfo(variable_name, &info));
  std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  std::vector<Data*> inputs = Args(
      info.datatype, 
      TensorShape(dims), 
      (size_t)0, 
      init);
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("IdentityIndexVariableInitializer", 
              UdfData(0), 
              UdfData(1), 
              UdfData(2), 
              UdfData(3));
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, '^' + variable_name, inputs, outputs, realcb);
}

void LocalClient::HashInitializer(const std::string& variable_name, 
                                  Initializer* init, 
                                  const LocalClient::Callback& cb) {
  VariableInfo info;
  CHECK_ASYNC(local_server_->GetVariableInfo(variable_name, &info));
  std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  size_t k = info.shape[0];
  dims[0] = k + 10 * sqrt(k) + 10;
  std::vector<Data*> inputs = Args(
      info.datatype, 
      TensorShape(dims), 
      std::unique_ptr<Initializer>(init));
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("HashVariableInitializer", UdfData(0), UdfData(1), UdfData(2));
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, '^' + variable_name, inputs, outputs, realcb);
}

void LocalClient::IsInitialized(const std::string& variable_name, 
                                bool* inited, 
                                const LocalClient::Callback& cb) {
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("IsInitialized");
  Callback realcb = [cb, outputs, inited](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    WrapperData<bool>* output_ptr = 
      dynamic_cast<WrapperData<bool>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor"));
      return;
    }

    *inited = output_ptr->Internal();
    cb(Status::Ok());
  };

  Process(udf, '^' + variable_name, {}, outputs, realcb);
}

void LocalClient::DensePull(const std::string& variable_name, 
                            Tensor* result, 
                            const LocalClient::Callback& cb) {
  std::vector<Data*> inputs = Args(false);
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("BuildDenseSlice", UdfData(0));
  UdfData udf_chain("SliceToTensor", UdfData("TransSlice", udf));
  Callback realcb = [this, cb, result, outputs, &variable_name](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on DensePull"));
      return;
    }

    WrapperData<Tensor>* output_ptr = dynamic_cast<WrapperData<Tensor>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor"));
      return;
    }

    *result = output_ptr->Internal();
    cb(Status::Ok());
  };

  Process(udf_chain, variable_name, inputs, outputs, realcb);
}

void LocalClient::DensePush(const std::string& variable_name, 
                            const std::string& updater, 
                            const std::vector<Data*>& data, 
                            const LocalClient::Callback& cb) {
  std::vector<Data*> inputs = Args(true);
  inputs.insert(inputs.end(), data.begin(), data.end());
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  std::vector<UdfData> updater_inputs = {UdfData("BuildDenseSlice", UdfData(0))};
  for (size_t i = 1; i < inputs.size(); i++) {
    updater_inputs.push_back(UdfData(i));
  }

  UdfData udf(updater, updater_inputs);
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, variable_name, inputs, outputs, realcb);
}

void LocalClient::SparsePull(const std::string& variable_name, 
                             const Tensor& ids, 
                             Tensor* result, 
                             const LocalClient::Callback& cb) {
  std::vector<Data*> inputs = Args(ids, false);
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("BuildSparseSlice", UdfData(0), UdfData(1));
  UdfData udf_chain("SliceToTensor", UdfData("TransSlice", udf));
  Callback realcb = [cb, result, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on SparsePull"));
      return;
    }

    WrapperData<Tensor>* output_ptr = 
      dynamic_cast<WrapperData<Tensor>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor"));
      return;
    }

    *result = output_ptr->Internal();
    cb(Status::Ok());
  };

  Process(udf_chain, variable_name, inputs, outputs, realcb);
}

void LocalClient::SparsePush(const std::string& variable_name, 
                             const Tensor& ids, 
                             const std::string& updater, 
                             const std::vector<Data*>& data, 
                             const LocalClient::Callback& cb) {
  std::vector<Data*> inputs = Args(ids, true);
  inputs.insert(inputs.end(), data.begin(), data.end());
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  std::vector<UdfData> updater_inputs = {
    UdfData("BuildSparseSlice", UdfData(0), UdfData(1))
  };
  for (size_t i = 2; i < inputs.size(); i++) {
    updater_inputs.push_back(UdfData(i));
  }

  UdfData udf(updater, updater_inputs);
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, variable_name, inputs, outputs, realcb);
}

void LocalClient::HashPull(const std::string& variable_name, 
                           const Tensor& ids, 
                           double filter_ratio,
                           Tensor* result, 
                           const LocalClient::Callback& cb) {
  std::vector<Data*> inputs = Args(ids, false, 1.0);
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2));
  UdfData udf_chain("SliceToTensor", UdfData("TransSlice", udf));
  Callback realcb = [cb, result, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on HashPull"));
      return;
    }

    WrapperData<Tensor>* output_ptr = 
      dynamic_cast<WrapperData<Tensor>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor"));
      return;
    }

    *result = output_ptr->Internal();
    cb(Status::Ok());
  };

  Process(udf_chain, variable_name, inputs, outputs, realcb);
}

void LocalClient::HashPush(const std::string& variable_name, 
                           const Tensor& ids, 
                           const std::string& updater, 
                           const std::vector<Data*>& data, 
                           const LocalClient::Callback& cb) {
  std::vector<Data*> inputs = Args(ids, true, 0.0);
  inputs.insert(inputs.end(), data.begin(), data.end());
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  std::vector<UdfData> updater_inputs = {
    UdfData("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2))
  };
  for (size_t i = 3; i < inputs.size(); i++) {
    updater_inputs.push_back(UdfData(i));
  }

  UdfData udf(updater, updater_inputs);
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, variable_name, inputs, outputs, realcb);
}

void LocalClient::Process(const UdfChain& udf, 
                          const std::string& var_name,
                          const std::vector<Data*>& datas,
                          std::vector<std::unique_ptr<Data>>* results,
                          const LocalClient::Callback& cb) {
  std::vector<Data*> outputs;
  Status st = local_server_->Process(udf.hash(), var_name, datas, &outputs);
  if (st.Code() == Status::kUdfNotRegistered) {
    st = local_server_->RegisterUdfChain(udf.BuildChainRegister());
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    st = local_server_->Process(udf.hash(), var_name, datas, &outputs);
  }

  results->reserve(outputs.size());
  for (Data* data: outputs) results->push_back(std::unique_ptr<Data>(data));
  cb(st);
  for (Data* data: datas) delete data;
}

} //namespace client
} //namespace ps
