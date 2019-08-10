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

#include "ps-plus/server/local_server.h"

#include "ps-plus/server/checkpoint_utils.h"
#include "ps-plus/common/serializer.h"
#include "ps-plus/common/file_system.h"
#include "ps-plus/common/hasher.h"

namespace ps {
namespace server {

LocalServer::LocalServer(const std::string& ckpt_path)
  : udf_chain_manager_(new UdfChainManager)
  , storage_manager_(new StorageManager) 
  , ckpt_path_(ckpt_path) {
}

Status LocalServer::Init() { return Status::Ok(); }

Status LocalServer::RegisterUdfChain(const UdfChainRegister& def) {
  return udf_chain_manager_->RegisterUdfChain(def);
}

Status LocalServer::Process(size_t udf, 
                            const std::string& variable_name,
                            const std::vector<Data*>& inputs,
                            std::vector<Data*>* outputs) {
  UdfContext ctx;
  Status st = RunUdfChain(udf, variable_name, inputs, &ctx);
  if (!st.IsOk()) return st;
  ctx.RemoveOutputDependency();
  outputs->insert(outputs->end(), ctx.Outputs().begin(), ctx.Outputs().end());
  return st;
}

Status LocalServer::RunUdfChain(size_t udf, 
                                const std::string& variable_name, 
                                const std::vector<Data*>& inputs, 
                                UdfContext* ctx) {
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

  return udf_chain->Process(ctx);
}

Status LocalServer::Restore(const std::string& ckpt_version) {
  std::vector<std::string> checkpoints;
  {
    std::unique_ptr<FileSystem::ReadStream> s;
    Status st = FileSystem::OpenReadStreamAny(ckpt_path_ + "/checkpoints", &s);
    // Ignore st fail when we use a fresh checkpoint dir
    if (st.IsOk()) {
      size_t size;
      PS_CHECK_STATUS(s->ReadRaw(&size));
      checkpoints.resize(size);
      for (size_t i = 0; i < size; i++) {
        PS_CHECK_STATUS(s->ReadStr(&checkpoints[i]));
      }
    }
  }

  std::string checkpoint = ckpt_version;
  if (checkpoint == "") {
    if (!checkpoints.empty()) {
      checkpoint = checkpoints.back();
    } else {
      return Status::Ok();
    }
  } else {
    bool found = false;
    for (auto& item: checkpoints) {
      if (checkpoint == item) {
        found = true;
        break;
      }
    }

    if (!found) {
      return Status::NotFound("Checkpoint Not Found : " + checkpoint);
    }
  }

  std::string real_ckpt_path = ckpt_path_ + "/" +  checkpoint;
  VariableInfoCollection info;
  PS_CHECK_STATUS(LoadCheckPointMeta(real_ckpt_path, &info));
  {
    std::lock_guard<std::mutex> lock(var_info_mutex_);
    for (auto& item: info.infos) {
      item.args[VariableInfo::ORIGIN_FILE_PATH] = real_ckpt_path;
      item.args[VariableInfo::ORIGIN_NAME] = item.name;
      size_t total = 0;
      for (const auto& part : item.parts) {
        total += part.size;
      }
      item.parts.clear();
      item.parts.push_back(ps::VariableInfo::Part{.server=0, .size=total});   
      var_infos_.insert(std::make_pair(item.name, item));
    }
  }

  storage_manager_->Internal().clear();
  CheckpointUtils ckpt_utils(info);
  return ckpt_utils.LoadVariables(info, 0, &storage_manager_->Internal());
}

Status LocalServer::LoadCheckPointMeta(const std::string& checkpoint,
                                       VariableInfoCollection* info) {
  std::unique_ptr<FileSystem::ReadStream> s;
  PS_CHECK_STATUS(FileSystem::OpenReadStreamAny(checkpoint + "/__meta__", &s));
  size_t server_cnt;

  {
    size_t infos_type;
    std::string infos_buf;
    PS_CHECK_STATUS(s->ReadRaw(&server_cnt));
    PS_CHECK_STATUS(s->ReadRaw(&infos_type));
    PS_CHECK_STATUS(s->ReadStr(&infos_buf));
    Data* info_wrapper;
    size_t len;
    serializer::MemGuard mem;
    serializer::Fragment frag(&infos_buf[0], infos_buf.size());
    PS_CHECK_STATUS(serializer::DeserializeAny<Data>(
                        infos_type, &frag, 0, 
                        &info_wrapper, 
                        &len, 
                        mem));
    std::unique_ptr<Data> info_wrapper_deleter(info_wrapper);
    WrapperData<VariableInfoCollection>* info_wrapper_converted = 
      dynamic_cast<WrapperData<VariableInfoCollection>*>(info_wrapper);
    if (info_wrapper_converted == nullptr) {
      return Status::Unknown("Variable Info Load Error");
    }

    info->infos = info_wrapper_converted->Internal().infos;
  }

  return Status::Ok();
}

// TODO: merge variable parts to one
Status LocalServer::GetVariableInfo(const std::string& var_name,
                                    VariableInfo* info) {
  std::lock_guard<std::mutex> lock(var_info_mutex_);
  auto it = var_infos_.find(var_name);
  if (it != var_infos_.end()) {
    *info = it->second;
    return Status::Ok();
  }

  return Status::Unknown("can't find var info for var:" + 
                         var_name);
}

Status LocalServer::RegisterVariable(const std::string& name, 
                                     const VariableInfo& info) {
  VariableInfo vi = info;
  if (vi.type == VariableInfo::kIndex) {
    size_t size = vi.shape.empty() ? 1 : (size_t)vi.shape[0];
    VariableInfo::Part part = {.server=0, .size=size};
    vi.parts.push_back(part);
  } else {
    VariableInfo::Part part = {.server=0, .size=Hasher::kTargetRange};
    vi.parts.push_back(part);
  }

  std::lock_guard<std::mutex> lock(var_info_mutex_);
  var_infos_[name] = vi;
  return Status::Ok();
}

Status LocalServer::Save(const std::string& ckpt_version) {
  VariableInfoCollection info;
  {
    std::lock_guard<std::mutex> lock(var_info_mutex_);
    for (auto it: var_infos_) {
      info.infos.push_back(it.second);
    }
  }

  {
    std::unique_ptr<FileSystem::WriteStream> s;
    PS_CHECK_STATUS(FileSystem::OpenWriteStreamAny(
                        ckpt_path_ + "/" + ckpt_version + 
                        "/__meta__", &s));
    {
      std::string infos_buf;
      size_t infos_type;
      std::unique_ptr<WrapperData<VariableInfoCollection>> info_wrapper(
          new WrapperData<VariableInfoCollection>);
      info_wrapper->Internal().infos = info.infos;
      std::vector<serializer::Fragment> frags;
      serializer::MemGuard mem;
      PS_CHECK_STATUS(serializer::SerializeAny<Data>(
                          info_wrapper.get(), &infos_type, &frags, mem));
      for (auto frag : frags) {
        infos_buf.append(frag.base, frag.size);
      }

      PS_CHECK_STATUS(s->WriteRaw(/*server_count*/(size_t)1));
      PS_CHECK_STATUS(s->WriteRaw(infos_type));
      PS_CHECK_STATUS(s->WriteStr(infos_buf));
    }
  }

  std::string real_ckpt_path = ckpt_path_ + "/" + ckpt_version;
  CheckpointUtils ckpt_utils(info);  
  PS_CHECK_STATUS(ckpt_utils.SaveVariables(0, real_ckpt_path, storage_manager_->Internal()));
  std::vector<std::string> checkpoints;
  {
    std::unique_ptr<FileSystem::ReadStream> s;
    Status st = FileSystem::OpenReadStreamAny(ckpt_path_ + "/checkpoints", &s);
    // Ignore st fail when we use a fresh checkpoint dir
    if (st.IsOk()) {
      size_t size;
      PS_CHECK_STATUS(s->ReadRaw(&size));
      checkpoints.resize(size);
      for (size_t i = 0; i < size; i++) {
        PS_CHECK_STATUS(s->ReadStr(&checkpoints[i]));
      }
    }
  }

  checkpoints.push_back(ckpt_version);
  {
    std::unique_ptr<FileSystem::WriteStream> s;
    PS_CHECK_STATUS(FileSystem::OpenWriteStreamAny(
                        ckpt_path_ + "/checkpoints.tmp", &s));
    size_t size = checkpoints.size();
    PS_CHECK_STATUS(s->WriteRaw(size));
    for (size_t i = 0; i < size; i++) {
      PS_CHECK_STATUS(s->WriteStr(checkpoints[i]));
    }
  }

  FileSystem::RemoveAny(ckpt_path_ + "/checkpoints");
  PS_CHECK_STATUS(FileSystem::RenameAny(ckpt_path_ + "/checkpoints.tmp", 
                                        ckpt_path_ + "/checkpoints"));
  return Status::Ok();;
}

}
}

