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

#include <future>

#include "ps-plus/scheduler/scheduler_service.h"
#include "ps-plus/service/seastar/lib/callback_closure.h"
#include "ps-plus/common/net_utils.h"
#include "ps-plus/common/reliable_kv.h"
#include "ps-plus/scheduler/scheduler_impl.h"

using ps::service::seastar::SeastarStatus;
using ps::service::seastar::CallBackClosure;
using ps::service::seastar::SeastarServerClientLib;

namespace ps {
namespace scheduler {

namespace {
  Status SeastarStatus2Status(const SeastarStatus& sst) {
    switch (sst.Code()) {
    case SeastarStatus::kNetworkError:
      return Status::NetworkError("network error");
      break;
    case SeastarStatus::kTimeout:
      return Status::Timeout("timeout");
      break;
    case SeastarStatus::kServerFuncNotFound:
      return Status::ServerFuncNotFound(sst.ToString());
      break;      
    case SeastarStatus::kServerSerializeFailed:
      return Status::ServerSerializeFailed(sst.ToString());
      break;      
    case SeastarStatus::kServerDeserializeFailed:
      return Status::ServerDeserializeFailed(sst.ToString());
      break;      
    case SeastarStatus::kClientSerializeFailed:
      return Status::ClientSerializeFailed(sst.ToString());
      break;      
    case SeastarStatus::kClientDeserializeFailed:
      return Status::ClientDeserializeFailed(sst.ToString());
      break;      
    default:
      return Status::Ok();
    }
  }

  Status GetNetworkStatus(const SeastarStatus& sst, const std::vector<ps::Data*>& datas) {
    if (!sst.Success()) {
      return SeastarStatus2Status(sst);
    }
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

Status SchedulerService::Start() {
  seastar_lib_.reset(new SeastarServerClientLib(port_, core_num_, core_num_, CLIENT_THREAD_NUM, bind_cores_));
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerGetVersion,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      GetVersion(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerSave,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      Save(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerRestore,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      Restore(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerRegisterServer,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      RegisterServer(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerGetClusterInfo,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      GetClusterInfo(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerUpdateVariableInfo,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      UpdateVariableInfo(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerTriggerStreamingDense,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      TriggerStreamingDense(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerTriggerStreamingSparse,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      TriggerStreamingSparse(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerTriggerStreamingHash,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      TriggerStreamingHash(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerSynchronizeEnter,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      SynchronizeEnter(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerSynchronizeLeave,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      SynchronizeLeave(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerAsynchronizeEnter,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      AsynchronizeEnter(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerWorkerBarrier,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      WorkerBarrier(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerWorkerReportFinish,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      WorkerReportFinish(inputs, outputs, done);
  });  
  seastar_lib_->RegisterServerFunc(func_ids::kSchedulerUpdateVariableVisitInfo,
      [this](const std::vector<ps::Data*>& inputs,
             std::vector<ps::Data*>* outputs,
             ps::service::seastar::DoneClosure* done) {
      UpdateVariableVisitInfo(inputs, outputs, done);
  });
  seastar_lib_->Start();

  NetUtils::GetDefaultIP(ip_);
  PS_CHECK_STATUS(ReliableKV::WriteAny(scheduler_kv_addr_, server_count_ + "^" + ip_ + ":" + std::to_string(port_)));
  return Status::Ok();
}

SchedulerService::~SchedulerService() {
  if (seastar_lib_ != nullptr) {
    seastar_lib_->Stop();
  }
}

void SchedulerService::SetServer(int server_type, int server_id, const std::string& server_addr) {
  seastar_lib_->Connect(server_offset_[server_type] + server_id, server_addr, true, true);      
}

//TODO: Add VariableInfos
void SchedulerService::ServerSave(
    int server_type,
    int server_id,
    Version version,
    const std::string& checkpoint,
    const std::vector<VariableInfo>& info,
    std::function<void(Status)> cb) {
  std::vector<Data*> datas = {
    new WrapperData<Version>(version),
    new WrapperData<std::string>(checkpoint),
    new WrapperData<VariableInfoCollection>(VariableInfoCollection{.infos = info})
  };
  seastar_lib_->Request(server_offset_[server_type] + server_id, func_ids::kServerSave, datas,
    new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<ps::Data*>& datas) {
    cb(GetNetworkStatus(sst, datas));
  }));
}

//TODO: Add VariableInfos
void SchedulerService::ServerRestore(
    int server_type,
    int server_id,
    Version version,
    const std::string& checkpoint,
    const std::vector<VariableInfo>& from,
    const std::vector<VariableInfo>& to,
    std::function<void(Status)> cb) {
  std::vector<Data*> datas = {
    new WrapperData<Version>(version),
    new WrapperData<std::string>(checkpoint),
    new WrapperData<VariableInfoCollection>(VariableInfoCollection{.infos = from}),
    new WrapperData<VariableInfoCollection>(VariableInfoCollection{.infos = to})
  }; seastar_lib_->Request(server_offset_[server_type] + server_id, func_ids::kServerRestore, datas,
    new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<ps::Data*>& datas) {
      cb(GetNetworkStatus(sst, datas));
  }));
}

void SchedulerService::ServerStreamingDenseVarName(
    int server_type,
    int server_id,
    Version version,
    std::function<void(Status, const DenseVarNames& vars)> cb) {
  std::vector<Data*> datas = {
    new WrapperData<Version>(version)
  };
  seastar_lib_->Request(server_offset_[server_type] + server_id, func_ids::kServerStreamingDenseVarName, datas,
    new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<ps::Data*>& datas) {
      Status st = GetNetworkStatus(sst, datas);
      if (!st.IsOk()) {
        cb(st, DenseVarNames());
        return;
      }
      if (datas.size() != 2) {
        cb(Status::Unknown("ServerStreamingDenseVarName Protocol Error, Size Error"), DenseVarNames());
        return;
      }
      WrapperData<DenseVarNames>* result = dynamic_cast<WrapperData<DenseVarNames>*>(datas[1]);
      if (result == nullptr) {
        cb(Status::Unknown("ServerStreamingDenseVarName Protocol Error, Type Error"), DenseVarNames());
        return;
      }
      cb(Status::Ok(), result->Internal());
  }));
}
void SchedulerService::ServerGatherStreamingDenseVar(
    int server_type,
    int server_id,
    Version version,
    const DenseVarNames& vars,
    std::function<void(Status, const DenseVarValues& vars)> cb) {
  std::vector<Data*> datas = {
    new WrapperData<Version>(version),
    new WrapperData<DenseVarNames>(vars)
  };
  seastar_lib_->Request(server_offset_[server_type] + server_id, func_ids::kServerGatherStreamingDenseVar, datas,
    new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<ps::Data*>& datas) {
      Status st = GetNetworkStatus(sst, datas);
      if (!st.IsOk()) {
        cb(st, DenseVarValues());
        return;
      }
      if (datas.size() != 2) {
        cb(Status::Unknown("ServerStreamingDenseVarName Protocol Error, Size Error"), DenseVarValues());
        return;
      }
      WrapperData<DenseVarValues>* result = dynamic_cast<WrapperData<DenseVarValues>*>(datas[1]);
      if (result == nullptr) {
        cb(Status::Unknown("ServerStreamingDenseVarName Protocol Error, Type Error"), DenseVarValues());
        return;
      }
      cb(Status::Ok(), result->Internal());
  }));
}

void SchedulerService::ServerTriggerStreamingSparse(
    int server_type,
    int server_id,
    Version version,
    std::function<void(Status)> cb) {
  std::vector<Data*> datas = {
    new WrapperData<Version>(version)
  };
  seastar_lib_->Request(server_offset_[server_type] + server_id, func_ids::kServerTriggerStreamingSparse, datas,
    new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<ps::Data*>& datas) {
      cb(GetNetworkStatus(sst, datas));
  }));
}

void SchedulerService::ServerTriggerStreamingHash(
    int server_type,
    int server_id,
    Version version,
    std::function<void(Status)> cb) {
  std::vector<Data*> datas = {
    new WrapperData<Version>(version)
  };
  seastar_lib_->Request(server_offset_[server_type] + server_id, func_ids::kServerTriggerStreamingHash, datas,
    new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<ps::Data*>& datas) {
      cb(GetNetworkStatus(sst, datas));
  }));
}

void SchedulerService::ModelServerFlush(
    int server_type,
    int server_id,
    Version version,
    std::function<void(Status)> cb) {
  std::vector<Data*> datas = {
    new WrapperData<Version>(version)
  };
  seastar_lib_->Request(server_offset_[server_type] + server_id, func_ids::kModelServerFlush, datas,
    new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<ps::Data*>& datas) {
      cb(GetNetworkStatus(sst, datas));
  }));
}

void SchedulerService::GetVersion(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done) {
  do {
    if (inputs.size() != 0) {
      outputs->push_back(new WrapperData<Status>(Status::ArgumentError("GetVersion: Need 0 inputs")));
      break;
    }

    Version result = impl_->GetVersion();
    if (result == kUnusedVersion) {
      outputs->push_back(new WrapperData<Status>(Status::NotReady("Server is not ready")));
      break;
    } else {
      outputs->push_back(new WrapperData<Status>(Status::Ok()));
      outputs->push_back(new WrapperData<Version>(result));
      break;
    }
  } while (0);

  done->Run();
}

void SchedulerService::Save(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 2) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService Save: Need 2 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<std::string>* data_ver = dynamic_cast<WrapperData<std::string>*>(inputs[1]);
  if (ver == nullptr || data_ver == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService Save: Input Type Error")));
    done->Run();
    return;
  }
  impl_->Save(ver->Internal(), data_ver->Internal(), [outputs, done](const Status& st) {
    outputs->push_back(new WrapperData<Status>(st));
    done->Run();
  });
}

void SchedulerService::Restore(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 2) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService Restore: Need 2 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<std::string>* data_ver = dynamic_cast<WrapperData<std::string>*>(inputs[1]);
  if (ver == nullptr || data_ver == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService Restore: Input Type Error")));
    done->Run();
    return;
  }
  impl_->Restore(ver->Internal(), data_ver->Internal(), [outputs, done, this](const Status& st) {
    outputs->push_back(new WrapperData<Status>(st));
    if (st.IsOk()) {
      outputs->push_back(new WrapperData<Version>(impl_->GetVersion()));
    }
    done->Run();
  });
}

void SchedulerService::RegisterServer(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 1) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService RegisterServer: Need 1 inputs")));
    done->Run();
    return;
  }
  WrapperData<ServerInfo>* info = dynamic_cast<WrapperData<ServerInfo>*>(inputs[0]);
  if (info == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService RegisterServer: Input Type Error")));
    done->Run();
    return;
  }
  Status st = impl_->RegisterServer(info->Internal());
  outputs->push_back(new WrapperData<Status>(st));
  done->Run();
}

void SchedulerService::GetClusterInfo(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 1) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService GetClusterInfo: Need 1 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  if (ver == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService GetClusterInfo: Input Type Error")));
    done->Run();
    return;
  }
  ClusterInfo info;
  Status st = impl_->GetClusterInfo(ver->Internal(), &info);
  outputs->push_back(new WrapperData<Status>(st));
  if (st.IsOk()) {
    outputs->push_back(new WrapperData<ClusterInfo>(info));
  }
  done->Run();
}

void SchedulerService::UpdateVariableInfo(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done) {
  if (inputs.size() == 0) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService UpdateVariableInfo: Need At least 1 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<VariableInfoCollection>* infos = dynamic_cast<WrapperData<VariableInfoCollection>*>(inputs[1]);
  if (ver == nullptr || infos == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService UpdateVariableInfo: Input Type Error")));
    done->Run();
    return;
  }
  VariableInfoCollection result;
  Status st = impl_->UpdateVariableInfo(ver->Internal(), infos->Internal().infos, &(result.infos));
  outputs->push_back(new WrapperData<Status>(st));
  if (st.IsOk()) {
    outputs->push_back(new WrapperData<VariableInfoCollection>(result));
  }
  done->Run();
}

void SchedulerService::TriggerStreamingDense(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 1) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService TriggerStreamingDense: Need 1 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  if (ver == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService TriggerStreamingDense: Input Type Error")));
    done->Run();
    return;
  }
  impl_->TriggerStreamingDense(ver->Internal(), [outputs, done](const Status& st) {
    outputs->push_back(new WrapperData<Status>(st));
    done->Run();
  });
}

void SchedulerService::TriggerStreamingSparse(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 1) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService TriggerStreamingSparse: Need 1 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  if (ver == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService TriggerStreamingSparse: Input Type Error")));
    done->Run();
    return;
  }
  impl_->TriggerStreamingSparse(ver->Internal(), [outputs, done](const Status& st) {
    outputs->push_back(new WrapperData<Status>(st));
    done->Run();
  });
}

void SchedulerService::TriggerStreamingHash(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 1) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService TriggerStreamingHash: Need 1 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  if (ver == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService TriggerStreamingHash: Input Type Error")));
    done->Run();
    return;
  }
  impl_->TriggerStreamingHash(ver->Internal(), [outputs, done](const Status& st) {
    outputs->push_back(new WrapperData<Status>(st));
    done->Run();
  });
}

void SchedulerService::AsynchronizeEnter(const std::vector<Data*>& inputs, std::vector<Data*>* outputs, ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 4) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService AsynchronizeEnter: Need 4 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<int>* id = dynamic_cast<WrapperData<int>*>(inputs[1]);
  WrapperData<int>* staleness = dynamic_cast<WrapperData<int>*>(inputs[2]);
  WrapperData<int>* worker_count = dynamic_cast<WrapperData<int>*>(inputs[3]);
  if (ver == nullptr || id == nullptr || staleness == nullptr || worker_count == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService AsynchronizeEnter: Input Type Error")));
    done->Run();
    return;
  }
  impl_->AsynchronizeEnter(ver->Internal(), id->Internal(),
                           staleness->Internal(),
                           worker_count->Internal(),
                           [outputs, done](const Status& st) {
    outputs->push_back(new WrapperData<Status>(st));
    done->Run();
  });
}

void SchedulerService::SynchronizeEnter(const std::vector<Data*>& inputs,
                                        std::vector<Data*>* outputs,
                                        ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 3) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService SynchronizeEnter: Need 3 inputs")));
    outputs->push_back(new WrapperData<int64_t>(-1));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<int>* id = dynamic_cast<WrapperData<int>*>(inputs[1]);
  WrapperData<int>* worker_count = dynamic_cast<WrapperData<int>*>(inputs[2]);
  if (ver == nullptr || id == nullptr || worker_count == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService SynchronizeEnter: Input Type Error")));
    outputs->push_back(new WrapperData<int64_t>(-1));
    done->Run();
    return;
  }
  impl_->SynchronizeEnter(ver->Internal(), id->Internal(),
                          worker_count->Internal(),
                          [outputs, done](int64_t token, const Status& st) {
    outputs->push_back(new WrapperData<Status>(st));
    outputs->push_back(new WrapperData<int64_t>(token));
    done->Run();
  });
}

void SchedulerService::SynchronizeLeave(const std::vector<Data*>& inputs,
                                        std::vector<Data*>* outputs,
                                        ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 3) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService SynchronizeLeave: Need 3 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<int>* id = dynamic_cast<WrapperData<int>*>(inputs[1]);
  WrapperData<int64_t>* token = dynamic_cast<WrapperData<int64_t>*>(inputs[2]);
  if (ver == nullptr || id == nullptr || token == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService SynchronizeLeave: Input Type Error")));
    done->Run();
    return;
  }
  impl_->SynchronizeLeave(ver->Internal(), id->Internal(), token->Internal(),
                          [outputs, done](const Status& st) {
    outputs->push_back(new WrapperData<Status>(st));
    done->Run();
  });
}

void SchedulerService::WorkerReportFinish(const std::vector<Data*>& inputs,
                                          std::vector<Data*>* outputs,
                                          ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 2) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService SynchronizeLeave: Need 3 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<int>* id = dynamic_cast<WrapperData<int>*>(inputs[1]);
  if (ver == nullptr || id == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService SynchronizeLeave: Input Type Error")));
    done->Run();
    return;
  }
  impl_->WorkerReportFinish(ver->Internal(), id->Internal(), [outputs, done](const Status& st) {
    outputs->push_back(new WrapperData<Status>(st));
    done->Run();
  });
}

void SchedulerService::WorkerBarrier(const std::vector<Data*>& inputs,
                                     std::vector<Data*>* outputs,
                                     ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 3) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService WorkerBarrier: Need 3 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<int>* id = dynamic_cast<WrapperData<int>*>(inputs[1]);
  WrapperData<int>* count = dynamic_cast<WrapperData<int>*>(inputs[2]);
  if (ver == nullptr || id == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService WorkerBarrier: Input Type Error")));
    done->Run();
    return;
  }
  impl_->WorkerBarrier(ver->Internal(), id->Internal(), count->Internal(), [outputs, done](const Status& st) {
    outputs->push_back(new WrapperData<Status>(st));
    done->Run();
  });
}

void SchedulerService::UpdateVariableVisitInfo(const std::vector<Data*>& inputs,
                                               std::vector<Data*>* outputs,
                                               ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 3) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService UpdateVariableVisitInfo: Need 3 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<std::string>* var_name = dynamic_cast<WrapperData<std::string>*>(inputs[1]);
  WrapperData<int64_t>* ids = dynamic_cast<WrapperData<int64_t>*>(inputs[2]);


  if (ver == nullptr || var_name == nullptr || ids == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("SchedulerService UpdateVariableVisitInfo: Input Type Error")));
    done->Run();
    return;
  }
  Status st = impl_->UpdateVariableVisitInfo(ver->Internal(), var_name->Internal(), ids->Internal());
  outputs->push_back(new WrapperData<Status>(st));
  done->Run();
}

int SchedulerService::GetServerSize(int server_type) {
  return server_offset_[server_type + 1] - server_offset_[server_type];
}
int SchedulerService::GetServerTotalSize() {
  return server_offset_.back();
}

int SchedulerService::GetServerTypeSize() {
  return server_offset_.size() - 1;
}

}
}

