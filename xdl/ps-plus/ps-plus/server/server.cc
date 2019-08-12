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

#include "ps-plus/server/server.h"
#include "ps-plus/server/checkpoint_utils.h"
#include "ps-plus/common/logging.h"

namespace ps {
namespace server {

Server::Server(size_t id, const StreamingModelArgs& streaming_model_args)
  : udf_chain_manager_(new UdfChainManager), 
    storage_manager_(new StorageManager),
    ver_(kUnusedVersion), id_(id),
    streaming_model_args_(streaming_model_args) {}

Status Server::Init() {
  PS_CHECK_STATUS(streaming_model_args_.Init());
  return Status::Ok();
}

Status Server::RegisterUdfChain(Version ver, const UdfChainRegister& def) {
  // Don't check version.
  return udf_chain_manager_->RegisterUdfChain(def);
}

Status Server::RunUdfChain(Version ver, size_t udf, const std::string& variable_name, const std::vector<Data*>& inputs, UdfContext* ctx) {
  QRWLocker lock(server_lock_, QRWLocker::kSimpleRead);
  if (ver != ver_) {
    return Status::VersionMismatch("RunUdfChain Version Mismatch");
  }
  UdfChain* udf_chain = udf_chain_manager_->GetUdfChain(udf);
  if (udf_chain == nullptr) {
    return Status::UdfNotRegistered("UdfChain not registered");
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    PS_CHECK_STATUS(ctx->SetData(i, inputs[i], false));
  }
  PS_CHECK_STATUS(ctx->SetStorageManager(storage_manager_.get()));
  PS_CHECK_STATUS(ctx->SetStreamingModelArgs(&streaming_model_args_));
  Variable* variable = nullptr;
  if (variable_name.empty()) {
    return Status::ArgumentError("Variable Name Should not be empty");
  }
  if (variable_name[0] == '^') {
    PS_CHECK_STATUS(ctx->SetVariableName(variable_name.substr(1)));
  } else {
    PS_CHECK_STATUS(storage_manager_->Get(variable_name, &variable));
    PS_CHECK_STATUS(ctx->SetVariableName(variable_name));
  }
  PS_CHECK_STATUS(ctx->SetVariable(variable));
  std::unique_ptr<QRWLocker> locker;
  if (variable != nullptr) {
    locker.reset(new QRWLocker(variable->VariableLock(), QRWLocker::kSimpleRead));
    ctx->SetLocker(locker.get());
  }
  ctx->SetServerLocker(&lock);
  Status ret = udf_chain->Process(ctx);
  return ret;
}

Status Server::Save(Version ver, const std::string& checkpoint, const VariableInfoCollection& info) {
  QRWLocker lock(server_lock_, QRWLocker::kSimpleRead);
  if (ver != ver_) {
    return Status::VersionMismatch("RunUdfChain Version Mismatch");
  }
  CheckpointUtils ckpt(info);
  return ckpt.SaveVariables(id_, checkpoint, storage_manager_->Internal());
}

Status Server::Restore(Version ver, const VariableInfoCollection& from, const VariableInfoCollection& to) {
  QRWLocker lock(server_lock_, QRWLocker::kWrite);
  ver_ = ver;
  storage_manager_->Internal().clear();
  CheckpointUtils ckpt(from);
  return ckpt.LoadVariables(to, id_, &storage_manager_->Internal());
}

Status Server::StreamingDenseVarName(Version ver, DenseVarNames* result) {
  {
    QRWLocker lock(server_lock_, QRWLocker::kSimpleRead);
    if (ver != ver_) {
      return Status::VersionMismatch("RunUdfChain Version Mismatch");
    }
  }
  if (streaming_model_args_.streaming_dense_model_addr.empty()) {
    return Status::Unknown("Streaming Dense Model is Disabled");
  }
  std::unordered_map<std::string, StreamingModelUtils::DenseLog> logs;
  StreamingModelUtils::GetDense(&logs);
  for (auto&& item : logs) {
    result->names.push_back(item.first);
  }
  return Status::Ok();
}

Status Server::GatherStreamingDenseVar(Version ver, const DenseVarNames& name, DenseVarValues* result) {
  QRWLocker lock(server_lock_, QRWLocker::kSimpleRead);
  if (ver != ver_) {
    return Status::VersionMismatch("RunUdfChain Version Mismatch");
  }
  for (auto&& item : name.names) {
    std::string var_name = item;
    Variable* var;
    Status st = storage_manager_->Get(var_name, &var);
    if (st.Code() == Status::kNotFound) {
      continue;
    }
    if (!st.IsOk()) {
      return st;
    }
    QRWLocker var_lock(var->VariableLock(), QRWLocker::kSimpleRead);
    DenseVarValues::DenseVarValue ret;
    WrapperData<size_t>* offset_slicer = dynamic_cast<WrapperData<size_t>*>(var->GetSlicer());
    if (offset_slicer == nullptr) {
      return Status::Unknown("Variable " + var_name + " is not a index variable");
    }
    ret.name = var_name;
    ret.offset = offset_slicer->Internal();
    ret.data = *var->GetData();
    result->values.push_back(ret);
  }
  return Status::Ok();
}

Status Server::TriggerStreamingSparse(Version ver, const int& server_id, const std::string& stream_version) {
  {
    QRWLocker lock(server_lock_, QRWLocker::kSimpleRead);
    if (ver != ver_) {
      return Status::VersionMismatch("RunUdfChain Version Mismatch");
    }
  }
  if (streaming_model_args_.streaming_hash_model_addr.empty()) {
    return Status::Unknown("Streaming Hash Model is Disabled");
  }
  if (streaming_model_args_.streaming_hash_model_writer == nullptr) {
    return Status::Unknown("Streaming Hash Model Writer connect error");
  }
  std::unordered_map<std::string, StreamingModelUtils::SparseLog> logs;
  StreamingModelUtils::GetSparse(&logs);
  std::vector<StreamingModelWriter::SparseModel> results;
  for (auto&& item : logs) {
    std::string var_name = item.first;
    auto&& log = item.second;
    Variable* var;
    PS_CHECK_STATUS(storage_manager_->Get(var_name, &var));
    QRWLocker var_lock(var->VariableLock(), QRWLocker::kSimpleRead);
    StreamingModelWriter::SparseModel ret;
    WrapperData<size_t>* offset_slicer = dynamic_cast<WrapperData<size_t>*>(var->GetSlicer());
    if (offset_slicer == nullptr) {
      return Status::Unknown("Variable " + var_name + " is not a index variable");
    }
    size_t offset = offset_slicer->Internal();
    ret.name = var_name;
    ret.data = *var->GetData();
    for (auto item : log.write_ids) {
      ret.ids.push_back(item);
      ret.offsets.push_back(item - offset);
    }
    results.emplace_back(std::move(ret));
  }
  return streaming_model_args_.streaming_sparse_model_writer->WriteSparseModel(results, stream_version, server_id);
}

Status Server::TriggerStreamingHash(Version ver, const int& server_id, const std::string& stream_version) {
  {
    QRWLocker lock(server_lock_, QRWLocker::kSimpleRead);
    if (ver != ver_) {
      return Status::VersionMismatch("RunUdfChain Version Mismatch");
    }
  }
  if (streaming_model_args_.streaming_hash_model_addr.empty()) {
    return Status::Unknown("Streaming Hash Model is Disabled");
  }
  if (streaming_model_args_.streaming_hash_model_writer == nullptr) {
    return Status::Unknown("Streaming Hash Model Writer connect error");
  }
  std::unordered_map<std::string, StreamingModelUtils::HashLog> logs;
  StreamingModelUtils::GetHash(&logs);
  std::vector<StreamingModelWriter::HashModel> results;
  for (auto&& item : logs) {
    std::string var_name = item.first;
    auto&& log = item.second;
    Variable* var;
    PS_CHECK_STATUS(storage_manager_->Get(var_name, &var));
    QRWLocker var_lock(var->VariableLock(), QRWLocker::kSimpleRead);
    StreamingModelWriter::HashModel ret;
    std::unique_ptr<HashMap>& hashmap = (dynamic_cast<WrapperData<std::unique_ptr<HashMap> >*>(var->GetSlicer()))->Internal();
    if (hashmap == nullptr) {
      return Status::Unknown("Variable " + var_name + " is not a hash variable");
    }
    std::vector<size_t> ids;
    int64_t* keys = new int64_t[log.write_ids.size()*2];
    int i = 0;
    for (auto it = log.write_ids.begin(); it != log.write_ids.end(); ++it){
        keys[2*i] = it->first;
        keys[2*i+1] = it->second;
        i++;
    }
    tbb::concurrent_vector<size_t> reids;
    size_t filter;
    size_t r = hashmap->Get(keys, log.write_ids.size(), false, 1.0, &ids, &reids, &filter);
    if (r != 0) {
      return Status::Unknown("Streaming Hash Model Get Hashmap error");
    }
    ret.name = var_name;
    ret.data = *var->GetData();
    for (size_t i = 0; i < ids.size(); i++) {
      if (ids[i] < 0) {
        continue;
      }
      std::pair<int64_t, int64_t> temp(keys[2*i], keys[2*i+1]);
      ret.ids.push_back(temp);
      ret.offsets.push_back(ids[i]);
    }
    ret.del_ids = std::vector<std::pair<int64_t, int64_t>>(log.del_ids.begin(), log.del_ids.end());
    results.emplace_back(std::move(ret));
    delete keys;
  }
  return streaming_model_args_.streaming_hash_model_writer->WriteHashModel(results, stream_version, server_id);
}

}
}

