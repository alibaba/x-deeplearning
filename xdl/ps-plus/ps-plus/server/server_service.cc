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

#include "ps-plus/server/server_service.h"

#include "ps-plus/common/logging.h"
#include "ps-plus/common/net_utils.h"
#include "ps-plus/common/reliable_kv.h"
#include "ps-plus/message/server_info.h"
#include "ps-plus/service/seastar/lib/callback_closure.h"
#include <thread>
#include <tuple>
#include <future>
#include <tuple>

using ps::service::seastar::SeastarStatus;
using ps::service::seastar::SeastarServerClientLib;

namespace ps {
namespace server {

namespace {
Status GetNetworkStatus(const std::vector<ps::Data*>& datas) {
  if (datas.empty()) {
    return Status::ArgumentError("status not found");
  }
  WrapperData<Status>* st = dynamic_cast<WrapperData<Status>*>(datas[0]);
  if (st == nullptr) {
    return Status::ArgumentError("status not found");
  } else {
    return st->Internal();
  }
}
}

ServerService::ServerService(
    const std::string& scheduler, int server_id,
    std::string streaming_dense_model_addr,
    std::string streaming_sparse_model_addr,
    std::string streaming_hash_model_addr,
    bool bind_cores) {
  port_ = NetUtils::GetAvailablePort();
  NetUtils::GetDefaultIP(ip_);
  server_id_ = server_id;
  core_num_ = NetUtils::GetAvailableCpuNum();
  stop_ = false;
  server_version_ = NewRandomVersion();
  scheduler_kv_addr_ = scheduler;
  streaming_dense_model_addr_ = streaming_dense_model_addr;
  streaming_sparse_model_addr_ = streaming_sparse_model_addr;
  streaming_hash_model_addr_ = streaming_hash_model_addr;
  bind_cores_ = bind_cores;
}

Status ServerService::Init() {
  StreamingModelArgs streaming_model_args;
  streaming_model_args.streaming_dense_model_addr = streaming_dense_model_addr_;
  streaming_model_args.streaming_sparse_model_addr = streaming_sparse_model_addr_;
  streaming_model_args.streaming_hash_model_addr = streaming_hash_model_addr_;
  server_.reset(new Server(server_id_, streaming_model_args));
  PS_CHECK_STATUS(server_->Init());
  lazy_queue_.reset(new ThreadPool(3));

  seastar_lib_.reset(new SeastarServerClientLib(port_, core_num_, core_num_, CLIENT_THREAD_NUM, bind_cores_));
  seastar_lib_->RegisterServerFunc(func_ids::kServerRegisterUdfChain, 
      [this](const std::vector<ps::Data*>& inputs, 
             std::vector<ps::Data*>* outputs, 
             ps::service::seastar::DoneClosure* done) {
    RegisterUdfChain(inputs, outputs);
    done->Run();
  });
  seastar_lib_->RegisterServerFunc(func_ids::kServerProcess, 
      [this](const std::vector<ps::Data*>& inputs, 
             std::vector<ps::Data*>* outputs, 
             ps::service::seastar::DoneClosure* done) {
    Process(inputs, outputs);
    done->Run();
  });
  seastar_lib_->RegisterServerFunc(func_ids::kServerSave, 
      [this](const std::vector<ps::Data*>& inputs, 
             std::vector<ps::Data*>* outputs, 
             ps::service::seastar::DoneClosure* done) {
    lazy_queue_->Schedule([=]{
      Save(inputs, outputs);
      done->Run();
    });
  });
  seastar_lib_->RegisterServerFunc(func_ids::kServerRestore, 
      [this](const std::vector<ps::Data*>& inputs, 
             std::vector<ps::Data*>* outputs, 
             ps::service::seastar::DoneClosure* done) {
    lazy_queue_->Schedule([=]{
      Restore(inputs, outputs);
      done->Run();
    });
  });
  seastar_lib_->RegisterServerFunc(func_ids::kServerStreamingDenseVarName, 
      [this](const std::vector<ps::Data*>& inputs, 
             std::vector<ps::Data*>* outputs, 
             ps::service::seastar::DoneClosure* done) {
    lazy_queue_->Schedule([=]{
      StreamingDenseVarName(inputs, outputs);
      done->Run();
    });
  });
  seastar_lib_->RegisterServerFunc(func_ids::kServerGatherStreamingDenseVar, 
      [this](const std::vector<ps::Data*>& inputs, 
             std::vector<ps::Data*>* outputs, 
             ps::service::seastar::DoneClosure* done) {
    lazy_queue_->Schedule([=]{
      GatherStreamingDenseVar(inputs, outputs);
      done->Run();
    });
  });
  seastar_lib_->RegisterServerFunc(func_ids::kServerTriggerStreamingSparse, 
      [this](const std::vector<ps::Data*>& inputs, 
             std::vector<ps::Data*>* outputs, 
             ps::service::seastar::DoneClosure* done) {
    lazy_queue_->Schedule([=]{
      TriggerStreamingSparse(inputs, outputs);
      done->Run();
    });
  });
  seastar_lib_->RegisterServerFunc(func_ids::kServerTriggerStreamingHash, 
      [this](const std::vector<ps::Data*>& inputs, 
             std::vector<ps::Data*>* outputs, 
             ps::service::seastar::DoneClosure* done) {
    lazy_queue_->Schedule([=]{
      TriggerStreamingHash(inputs, outputs);
      done->Run();
    });
  });
  seastar_lib_->Start();

  // TODO: move cpus
  std::vector<std::tuple<int64_t, std::string>> server_addrs = { std::make_tuple(0, "") };
  register_server_loop_.reset(new std::thread([this]{RegisterServer();}));

  return Status::Ok();
}

ServerService::~ServerService() {
  stop_ = true;
  if (register_server_loop_ != nullptr) {
    register_server_loop_->join();
  }
  if (seastar_lib_ != nullptr) {
    seastar_lib_->Stop();
  }
}

void ServerService::RegisterUdfChain(const std::vector<Data*>& inputs, std::vector<Data*>* outputs) {
  if (inputs.size() != 2) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("RegisterUdfChainFunc: Need 2 inputs")));
    return;
  }

  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<UdfChainRegister>* def = dynamic_cast<WrapperData<UdfChainRegister>*>(inputs[1]);
  if (ver == nullptr || def == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("RegisterUdfChainFunc: Input Type Error")));
    return;
  }
  Status st = server_->RegisterUdfChain(ver->Internal(), def->Internal());
  outputs->push_back(new WrapperData<Status>(st));
  return;
}

void ServerService::Process(const std::vector<Data*>& inputs, std::vector<Data*>* outputs) {
  if (inputs.size() < 3) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("ProcessFunc: Need at least 3 inputs")));
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<size_t>* udf = dynamic_cast<WrapperData<size_t>*>(inputs[1]);
  WrapperData<std::string>* variable_name = dynamic_cast<WrapperData<std::string>*>(inputs[2]);
  if (ver == nullptr || udf == nullptr || variable_name == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("ProcessFunc: Input Type Error")));
    return;
  }
  std::vector<Data*> in(inputs.begin() + 3, inputs.end());
  UdfContext ctx;
  Status st = server_->RunUdfChain(ver->Internal(), udf->Internal(), variable_name->Internal(), in, &ctx);
  if (!st.IsOk()) {
    outputs->push_back(new WrapperData<Status>(st));
    return;
  }
  ctx.RemoveOutputDependency();
  outputs->push_back(new WrapperData<Status>(st));
  outputs->insert(outputs->end(), ctx.Outputs().begin(), ctx.Outputs().end());
  return;
}

void ServerService::Save(const std::vector<Data*>& inputs, std::vector<Data*>* outputs) {
  if (inputs.size() != 3) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SaveFunc: Need 3 inputs")));
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<std::string>* checkpoint = dynamic_cast<WrapperData<std::string>*>(inputs[1]);
  WrapperData<VariableInfoCollection>* info = dynamic_cast<WrapperData<VariableInfoCollection>*>(inputs[2]);
  if (ver == nullptr || checkpoint == nullptr || info == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SaveFunc: Input Type Error")));
    return;
  }
  LOG(INFO) << "Saving Checkpoint " << checkpoint->Internal().c_str();
  Status st = server_->Save(ver->Internal(), checkpoint->Internal(), info->Internal());
  outputs->push_back(new WrapperData<Status>(st));
  LOG(INFO) << "Saving Checkpoint Done " << checkpoint->Internal().c_str();
  return;
}

void ServerService::Restore(const std::vector<Data*>& inputs, std::vector<Data*>* outputs) {
  if (inputs.size() != 3) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("RestoreFunc: Need 3 inputs")));
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<VariableInfoCollection>* from = dynamic_cast<WrapperData<VariableInfoCollection>*>(inputs[1]);
  WrapperData<VariableInfoCollection>* to = dynamic_cast<WrapperData<VariableInfoCollection>*>(inputs[2]);
  if (ver == nullptr || from == nullptr || to == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("RestoreFunc: Input Type Error")));
    return;
  }
  Status st = server_->Restore(ver->Internal(), from->Internal(), to->Internal());
  outputs->push_back(new WrapperData<Status>(st));
  return;
}

void ServerService::StreamingDenseVarName(const std::vector<Data*>& inputs, std::vector<Data*>* outputs) {
  if (inputs.size() != 1) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("StreamingDenseVarNameFunc: Need 1 inputs")));
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  if (ver == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("StreamingDenseVarNameFunc: Input Type Error")));
    return;
  }
  DenseVarNames names;
  Status st = server_->StreamingDenseVarName(ver->Internal(), &names);
  outputs->push_back(new WrapperData<Status>(st));
  if (st.IsOk()) {
    outputs->push_back(new WrapperData<DenseVarNames>(std::move(names)));
  }
  return;
}

void ServerService::GatherStreamingDenseVar(const std::vector<Data*>& inputs, std::vector<Data*>* outputs) {
  if (inputs.size() != 2) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("GatherStreamingDenseVarFunc: Need 2 inputs")));
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<DenseVarNames>* names = dynamic_cast<WrapperData<DenseVarNames>*>(inputs[1]);
  if (ver == nullptr || names == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("GatherStreamingDenseVarFunc: Input Type Error")));
    return;
  }
  DenseVarValues vals;
  Status st = server_->GatherStreamingDenseVar(ver->Internal(), names->Internal(), &vals);
  outputs->push_back(new WrapperData<Status>(st));
  if (st.IsOk()) {
    outputs->push_back(new WrapperData<DenseVarValues>(std::move(vals)));
  }
  return;
}

void ServerService::TriggerStreamingSparse(const std::vector<Data*>& inputs, std::vector<Data*>* outputs) {
  if (inputs.size() != 2) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("TriggerStreamingSparseFunc: Need 2 inputs")));
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<std::string>* stream_version = dynamic_cast<WrapperData<std::string>*>(inputs[1]);
  if ((ver == nullptr) || (stream_version == nullptr)){
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("TriggerStreamingSparseFunc: Input Type Error")));
    return;
  }
  Status st = server_->TriggerStreamingSparse(ver->Internal(), server_id_, stream_version->Internal());
  outputs->push_back(new WrapperData<Status>(st));
  return;
}

void ServerService::TriggerStreamingHash(const std::vector<Data*>& inputs, std::vector<Data*>* outputs) {
  if (inputs.size() != 2) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("TriggerStreamingHashFunc: Need 2 inputs")));
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<std::string>* stream_version = dynamic_cast<WrapperData<std::string>*>(inputs[1]);
  if ((ver == nullptr) || (stream_version == nullptr)){
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("TriggerStreamingHashFunc: Input Type Error")));
    return;
  }
  Status st = server_->TriggerStreamingHash(ver->Internal(), server_id_, stream_version->Internal());
  outputs->push_back(new WrapperData<Status>(st));
  return;
}

void ServerService::RegisterServer() {
  std::string old_scheduler_addr;
  while (!stop_) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::string scheduler_addr;
    Status st = ReliableKV::ReadAny(scheduler_kv_addr_, &scheduler_addr);
    if (!st.IsOk()) {
      LOG(WARNING) << "Cannot Get Scheduler Addr[" << scheduler_kv_addr_ << "]: Status[" << st.ToString();
      continue;
    }

    size_t pos = scheduler_addr.find('^');
    if (pos == std::string::npos) {
      LOG(WARNING) << "Cannot Get Scheduler Addr[" << scheduler_kv_addr_ <<
                "]: Store[" << scheduler_addr << "]";
    }
    scheduler_addr = scheduler_addr.substr(pos + 1);

    if (scheduler_addr != old_scheduler_addr) {
      seastar_lib_->Connect(0, scheduler_addr);
      old_scheduler_addr = scheduler_addr;
    }

    std::promise<Status> result;
    std::vector<Data*> request_datas = {
      new WrapperData<ps::ServerInfo>(0, server_id_, server_version_, ip_, port_)
    };
    seastar_lib_->Request(0, func_ids::kSchedulerRegisterServer, request_datas,
        new ps::service::seastar::CallBackClosure([&result](const SeastarStatus& sst, const std::vector<ps::Data*>& datas) {
          if (!sst.Success()) {
            LOG(ERROR) << "register server failed, error: " << sst.ToString();
            result.set_value(Status::NetworkError("Scheduler Error"));
          } else {
            LOG(INFO) << "register server result: " << GetNetworkStatus(datas).ToString();
            result.set_value(GetNetworkStatus(datas));
          }
    }));
    result.get_future().wait();
    // TODO: process errors
  }
}

}
}

