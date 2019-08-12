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

#include "ps-plus/common/status.h"
#include "ps-plus/client/client_wrapper_impl.h"
#include "ps-plus/common/reliable_kv.h"
#include <future>
#include <iostream>

using ps::service::seastar::CallBackClosure;
using ps::service::seastar::SeastarClientLib;
using ps::service::seastar::EventClientLib;
using ps::service::seastar::SeastarStatus;

namespace ps {
namespace client {

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

ClientWrapperImpl::ClientLib* ClientWrapperImpl::client_lib_singleton_ = nullptr;

Status ClientWrapperImpl::ConnectToCluster(const std::string& addr) {
  PS_CHECK_STATUS(CreateServerLib());
  PS_CHECK_STATUS(ConnectToScheduler(addr));
  PS_CHECK_STATUS(WaitForReady());
  PS_CHECK_STATUS(ConnectToServers());
  return Status::Ok();
}

void ClientWrapperImpl::UpdateVariableVisitInfo(const std::string& name, int64_t id_num, const Callback& cb) {
  std::vector<Data*> request_datas;

  WrapperData<Version>* version_data = new WrapperData<Version>(scheduler_version_); 
  request_datas.push_back(version_data);

  WrapperData<std::string>* var_data = new WrapperData<std::string>(name);
  request_datas.push_back(var_data);

  WrapperData<int64_t>* ids_data = new WrapperData<int64_t>(id_num);
  request_datas.push_back(ids_data);

  CallBackClosure* cb_closure = new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<Data*>& response){
    cb(GetNetworkStatus(sst, response));
  });

  client_lib_->Request(0, func_ids::kSchedulerUpdateVariableVisitInfo, request_datas, cb_closure);
}
void ClientWrapperImpl::UpdateVariableInfo(const std::vector<VariableInfo>& input, 
                                       std::vector<VariableInfo>* output, 
                                       const Callback& cb) {
  std::vector<Data*> request_datas;

  WrapperData<Version>* version_data = new WrapperData<Version>(scheduler_version_); 
  request_datas.push_back(version_data);

  WrapperData<VariableInfoCollection>* infos = 
    new WrapperData<VariableInfoCollection>(VariableInfoCollection({.infos = input}));
  request_datas.push_back(infos);

  CallBackClosure* cb_closure = new CallBackClosure([output, cb](const SeastarStatus& sst, const std::vector<Data*>& response) {
    Status st = GetNetworkStatus(sst, response);
    if (!st.IsOk()) {
      cb(st);
      return;
    }
    if (response.size() != 2) {
      cb(Status::ArgumentError("UpdateVariableInfo: Response should be 2 datas"));
      return;
    }
    WrapperData<VariableInfoCollection>* info_wrapper = dynamic_cast<WrapperData<VariableInfoCollection>*>(response[1]);
    if (info_wrapper == nullptr) {
      cb(Status::ArgumentError("UpdateVariableInfo: Response should be VariableInfoCollection"));
      return;
    }
    *output = info_wrapper->Internal().infos;
    cb(Status::Ok());
  });

  client_lib_->Request(0, func_ids::kSchedulerUpdateVariableInfo,
    request_datas, cb_closure);
}

void ClientWrapperImpl::Process(const std::string& var_name, 
                                size_t server_id, 
                                size_t udf_id, 
                                const std::vector<Data*>& input, 
                                std::vector<Data*>* output, 
                                const Callback& cb) {
  std::vector<Data*> request_datas;
  
  WrapperData<Version>* version_data = new WrapperData<Version>(scheduler_version_); 
  request_datas.push_back(version_data);
  
  WrapperData<size_t>* udf_data = new WrapperData<size_t>(udf_id);
  request_datas.push_back(udf_data);
  
  WrapperData<std::string>* var_data = new WrapperData<std::string>(var_name);
  request_datas.push_back(var_data);

  request_datas.insert(request_datas.end(), input.begin(), input.end());
  CallBackClosure* cb_closure = new CallBackClosure([this, output, cb, var_name, server_id, udf_id, 
						     version_data, udf_data, var_data](const SeastarStatus& sst, const std::vector<Data*>& response) {
    std::unique_ptr<WrapperData<Version>> version_deleter(version_data);
    std::unique_ptr<WrapperData<size_t>> udf_deleter(udf_data);
    std::unique_ptr<WrapperData<std::string> > var_deleter(var_data);
    Status st = GetNetworkStatus(sst, response);
    if (!st.IsOk()) {
      cb(st);
      return;
    }
    (*output) = std::vector<Data*>(response.begin() + 1, response.end());
    cb(Status::Ok());
  });

  client_lib_->Request(server_id + offset_[0], func_ids::kServerProcess, request_datas, cb_closure, false);
}

void ClientWrapperImpl::RegisterUdf(size_t server_id, const UdfChain& def, const Callback& cb) {
  std::vector<Data*> request_datas;
  
  WrapperData<Version>* version_data = new WrapperData<Version>(scheduler_version_); 
  request_datas.push_back(version_data);
  
  WrapperData<UdfChainRegister>* udf_data = new WrapperData<UdfChainRegister>(def.BuildChainRegister());

  request_datas.push_back(udf_data);

  CallBackClosure* cb_closure = new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<Data*>& response) {
    cb(GetNetworkStatus(sst, response));
  });

  client_lib_->Request(server_id + offset_[0], func_ids::kServerRegisterUdfChain, request_datas, cb_closure);
}

void ClientWrapperImpl::Save(const std::string& version, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<std::string>(version)
  };

  CallBackClosure* cb_closure = new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<Data*>& response) {
    cb(GetNetworkStatus(sst, response));
  });

  client_lib_->Request(0, func_ids::kSchedulerSave, request_datas, cb_closure);
}

void ClientWrapperImpl::Restore(const std::string& version, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<std::string>(version)
  };

  CallBackClosure* cb_closure = new CallBackClosure([cb, this](const SeastarStatus& sst, const std::vector<Data*>& response) {
    Status st = GetNetworkStatus(sst, response);
    if (!st.IsOk()) {
      cb(st);
      return;
    }
    if (response.size() != 2) {
      cb(Status::Unknown("Restore Protocol Error"));
      return;
    }
    WrapperData<Version>* new_ver = dynamic_cast<WrapperData<Version>*>(response[1]);
    if (new_ver == nullptr) {
      cb(Status::Unknown("Restore Protocol Error"));
      return;
    }
    scheduler_version_ = new_ver->Internal();
    cb(Status::Ok());
  });

  client_lib_->Request(0, func_ids::kSchedulerRestore, request_datas, cb_closure);
}

Status ClientWrapperImpl::InitGlobalQueue(
    const std::string& name, 
    const std::vector<std::string>& paths, 
    size_t epochs, 
    bool epoch_isolate) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<std::string>(name),
    new WrapperData<std::vector<std::string> >(paths),
    new WrapperData<size_t>(epochs),
    new WrapperData<bool>(epoch_isolate)
  };

  std::promise<Status> p;
  CallBackClosure* cb_closure =
    new CallBackClosure([&p](const SeastarStatus& sst,
                             const std::vector<Data*>& response) {
      Status st = GetNetworkStatus(sst, response);
      p.set_value(st);
    });

  client_lib_->Request(0, func_ids::kSchedulerInitGlobalFileQueue,
                       request_datas, cb_closure);
  return p.get_future().get();
}

Status ClientWrapperImpl::GetNextFile(
    const std::string& name, 
    size_t worker_id, 
    std::string* path, 
    size_t* begin, 
    size_t* epoch) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<std::string>(name),
    new WrapperData<size_t>(worker_id)
  };

  std::promise<Status> p;
  CallBackClosure* cb_closure = 
    new CallBackClosure([&p, path, begin, epoch, this](
                            const SeastarStatus& sst, 
                            const std::vector<Data*>& response) {
    Status st = GetNetworkStatus(sst, response);
    if (!st.IsOk()) {
      p.set_value(st);
      return;
    }

    if (response.size() != 4) {
      p.set_value(Status::Unknown("response data not match"));
      return;
    }

    WrapperData<std::string>* path_data = dynamic_cast<WrapperData<std::string>* >(response[1]);
    WrapperData<size_t>* begin_data = dynamic_cast<WrapperData<size_t>* >(response[2]);    
    WrapperData<size_t>* epoch_data = dynamic_cast<WrapperData<size_t>* >(response[3]);    
    if (path_data == nullptr || begin_data == nullptr || epoch_data == nullptr) {
      p.set_value(Status::Unknown("reponse data type not match"));
      return;
    }

    *path = path_data->Internal();
    *begin = begin_data->Internal();
    *epoch = epoch_data->Internal();
    p.set_value(Status::Ok());
  });

  client_lib_->Request(0, func_ids::kSchedulerGetNextFile,
                       request_datas, cb_closure);
  return p.get_future().get();
}

Status ClientWrapperImpl::ReportWorkerState(
    const std::string& name,
    size_t worker_id, 
    const std::vector<WorkerState>& worker_states) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<std::string>(name),
    new WrapperData<size_t>(worker_id),
    new WrapperData<std::vector<WorkerState> >(worker_states)
  };

  std::promise<Status> p;
  CallBackClosure* cb_closure =
    new CallBackClosure([&p](const SeastarStatus& sst,
                             const std::vector<Data*>& response) {
      Status st = GetNetworkStatus(sst, response);
      p.set_value(st);
    });

  client_lib_->Request(0, func_ids::kSchedulerReportWorkerState,
                       request_datas, cb_closure);
  return p.get_future().get();
}

Status ClientWrapperImpl::RestoreWorkerState(
    const std::string& name,
    size_t worker_id) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<std::string>(name),
    new WrapperData<size_t>(worker_id)
  };

  std::promise<Status> p;
  CallBackClosure* cb_closure =
    new CallBackClosure([&p](const SeastarStatus& sst,
                             const std::vector<Data*>& response) {
      Status st = GetNetworkStatus(sst, response);
      p.set_value(st);
    });

  client_lib_->Request(0, func_ids::kSchedulerRestoreWorkerState,
                       request_datas, cb_closure);
  return p.get_future().get();
}

void ClientWrapperImpl::ModelServerForward(int server_type, int server_id, const Tensor& ids, std::unique_ptr<Tensor>* rst, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<Tensor>(ids)
  };
  CallBackClosure* cb_closure = new CallBackClosure([cb, rst](const SeastarStatus& sst, const std::vector<Data*>& response) {
    Status st = GetNetworkStatus(sst, response);
    if (!st.IsOk()) {
      cb(st);
      return;
    }
    if (response.size() != 2) {
      cb(Status::ArgumentError("ModelServerForward: Response should be 2 datas"));
      return;
    }
    WrapperData<Tensor>* rst_wrapper = dynamic_cast<WrapperData<Tensor>*>(response[1]);
    if (rst_wrapper == nullptr) {
      cb(Status::ArgumentError("ModelServerForward: Response should be Tensor"));
      return;
    }
    rst->reset(new Tensor(rst_wrapper->Internal()));
    cb(Status::Ok());
  });
  client_lib_->Request(server_id + offset_[server_type], func_ids::kModelServerForward,
    request_datas, cb_closure);
}

void ClientWrapperImpl::ModelServerBackward(int server_type, int server_id, const Tensor& ids, const Tensor& grads, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<Tensor>(ids),
    new WrapperData<Tensor>(grads)
  };
  CallBackClosure* cb_closure = new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<Data*>& response) {
    Status st = GetNetworkStatus(sst, response);
    cb(st);
  });
  client_lib_->Request(server_id + offset_[server_type], func_ids::kModelServerBackward,
    request_datas, cb_closure);
}

void ClientWrapperImpl::TriggerStreamingModelDense(const std::string& stream_ver, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<std::string>(stream_ver)
  };

  CallBackClosure* cb_closure = new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<Data*>& response) {
    cb(GetNetworkStatus(sst, response));
  });

  client_lib_->Request(0, func_ids::kSchedulerTriggerStreamingDense, request_datas, cb_closure);
}

void ClientWrapperImpl::TriggerStreamingModelSparse(const std::string& stream_ver, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<std::string>(stream_ver)
  };

  CallBackClosure* cb_closure = new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<Data*>& response) {
    cb(GetNetworkStatus(sst, response));
  });

  client_lib_->Request(0, func_ids::kSchedulerTriggerStreamingSparse, request_datas, cb_closure);
}

void ClientWrapperImpl::TriggerStreamingModelHash(const std::string& stream_ver, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<std::string>(stream_ver)
  };

  CallBackClosure* cb_closure = new CallBackClosure([cb](const SeastarStatus& sst, const std::vector<Data*>& response) {
    cb(GetNetworkStatus(sst, response));
  });

  client_lib_->Request(0, func_ids::kSchedulerTriggerStreamingHash, request_datas, cb_closure);
}

void ClientWrapperImpl::AsynchronizeEnter(int id, int staleness, int worker_count, const Callback& cb) {
  if (staleness < 0) {
    cb(Status::ArgumentError("Staleness should not be negative"));
  }
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<int>(id),
    new WrapperData<int>(staleness),
    new WrapperData<int>(worker_count),
  };
  CallBackClosure* cb_closure =
    new CallBackClosure([cb](const SeastarStatus& sst,
                             const std::vector<Data*>& response) {
      cb(GetNetworkStatus(sst, response));
    });
  client_lib_->Request(0, func_ids::kSchedulerAsynchronizeEnter,
                       request_datas, cb_closure);
}

void ClientWrapperImpl::SynchronizeEnter(int id, int worker_count, int64_t* token, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<int>(id),
    new WrapperData<int>(worker_count),
  };
  CallBackClosure* cb_closure =
    new CallBackClosure([token, cb](const SeastarStatus& sst,
                             const std::vector<Data*>& response) {
      Status st = GetNetworkStatus(sst, response);
      if (!st.IsOk()) {
        cb(st);
        return;
      }
      if (token) {
        WrapperData<int64_t>* res = dynamic_cast<WrapperData<int64_t>*>(response[1]);
        *token = res->Internal();
      }
      cb(Status::Ok());
    });
  client_lib_->Request(0, func_ids::kSchedulerSynchronizeEnter,
                       request_datas, cb_closure);
}

void ClientWrapperImpl::SynchronizeLeave(int id, int64_t token, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<int>(id),
    new WrapperData<int64_t>(token),
  };
  CallBackClosure* cb_closure =
    new CallBackClosure([cb](const SeastarStatus& sst,
                             const std::vector<Data*>& response) {
      cb(GetNetworkStatus(sst, response));
    });
  client_lib_->Request(0, func_ids::kSchedulerSynchronizeLeave,
                       request_datas, cb_closure);
}

void ClientWrapperImpl::WorkerReportFinish(int id, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<int>(id)
  };
  CallBackClosure* cb_closure =
    new CallBackClosure([cb](const SeastarStatus& sst,
                             const std::vector<Data*>& response) {
      cb(GetNetworkStatus(sst, response));
    });
  client_lib_->Request(0, func_ids::kSchedulerWorkerReportFinish,
                       request_datas, cb_closure);
}

void ClientWrapperImpl::GetWorkerFinishCount(int64_t* count, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_)
  };
  CallBackClosure* cb_closure =
    new CallBackClosure([count, cb](const SeastarStatus& sst,
                                    const std::vector<Data*>& response) {
      Status st = GetNetworkStatus(sst, response);
      if (!st.IsOk()) {
        cb(st);
        return;
      }
      if (count) {
        WrapperData<int64_t>* res = dynamic_cast<WrapperData<int64_t>*>(response[1]);
        *count = res->Internal();
      }
      cb(Status::Ok());
    });
  client_lib_->Request(0, func_ids::kSchedulerGetWorkerFinishCount,
                       request_datas, cb_closure);
}    

void ClientWrapperImpl::WorkerBarrier(int id, int worker_count, const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<int>(id),
    new WrapperData<int>(worker_count)    
  };
  CallBackClosure* cb_closure =
    new CallBackClosure([cb](const SeastarStatus& sst,
                             const std::vector<Data*>& response) {
      cb(GetNetworkStatus(sst, response));
    });
  client_lib_->Request(0, func_ids::kSchedulerWorkerBarrier,
                       request_datas, cb_closure);
}

void ClientWrapperImpl::WorkerBarrierV2(
    int barrier_id, 
    int task_id, 
    int task_num,
    int token,
    const Callback& cb) {
  std::vector<Data*> request_datas = {
    new WrapperData<Version>(scheduler_version_),
    new WrapperData<int>(barrier_id),
    new WrapperData<int>(task_id),
    new WrapperData<int>(task_num),
    new WrapperData<int>(token)    
  };
  CallBackClosure* cb_closure =
    new CallBackClosure([cb](const SeastarStatus& sst,
                             const std::vector<Data*>& response) {
      cb(GetNetworkStatus(sst, response));
    });
  client_lib_->Request(0, func_ids::kSchedulerWorkerBarrierV2,
                       request_datas, cb_closure);
}

Status ClientWrapperImpl::CreateServerLib() {
  if (client_lib_singleton_ == nullptr) {
    std::vector<std::tuple<int64_t, std::string>> server_addrs = {};
    client_lib_singleton_ = new ClientLib(server_addrs, 100, std::thread::hardware_concurrency());
    client_lib_ = client_lib_singleton_;
    client_lib_->Start();
  } else {
    client_lib_ = client_lib_singleton_;
  }
  return Status::Ok();
}

Status ClientWrapperImpl::ConnectToScheduler(const std::string& addr) {
  std::string scheduler_addr;
  for (int i = 10, k = 4; ; i--, k = k + k / 4) {
    Status st = ReliableKV::ReadAny(addr, &scheduler_addr);
    if (st.IsOk()) {
      break;
    } else if (i <= 0) {
      return st;
    }
    std::this_thread::sleep_for(std::chrono::seconds(k));
  }

  size_t pos = scheduler_addr.find("^");
  if (pos == std::string::npos) {
    return Status::ArgumentError("invalid scheduler addr");
  } else {
    client_lib_->Connect(0, scheduler_addr.substr(pos + 1));
  }

  return Status::Ok();
}

Status ClientWrapperImpl::WaitForReady() {
  for (int i = 20; ; i--) {
    std::promise<Status> st_promise;
    Version ver;
    client_lib_->Request(0, func_ids::kSchedulerGetVersion, {}, new CallBackClosure([&](const SeastarStatus& sst, const std::vector<Data*>& response) {
      Status st = GetNetworkStatus(sst, response);
      if (!st.IsOk()) {
        st_promise.set_value(st);
        return;
      }
      if (response.size() != 2) {
        st_promise.set_value(Status::ArgumentError("WaitForReady: Response should be 2 datas"));
        return;
      }
      WrapperData<Version>* ver_wrapper = dynamic_cast<WrapperData<Version>*>(response[1]);
      if (ver_wrapper == nullptr) {
        st_promise.set_value(Status::ArgumentError("WaitForReady: Response should be Version"));
        return;
      }
      ver = ver_wrapper->Internal();
      st_promise.set_value(Status::Ok());
    }));
    std::future<Status> st_future = st_promise.get_future();
    st_future.wait();
    Status st = st_future.get();
    if (st.IsOk()) {
      scheduler_version_ = ver;
      return Status::Ok();
    } else if (i <= 0) {
      return st;
    }
    std::this_thread::sleep_for(std::chrono::seconds(6));
  }
}

Status ClientWrapperImpl::ConnectToServers() {
  std::promise<Status> st_promise;
  ClusterInfo info;
  client_lib_->Request(0,
      func_ids::kSchedulerGetClusterInfo,
      {new WrapperData<Version>(scheduler_version_)},
      new CallBackClosure([&](const SeastarStatus& sst, const std::vector<Data*>& response) {
    Status st = GetNetworkStatus(sst, response);
    if (!st.IsOk()) {
      st_promise.set_value(st);
      return;
    }
    if (response.size() != 2) {
      st_promise.set_value(Status::ArgumentError("ConnectToServers: Response should be 2 datas"));
      return;
    }
    WrapperData<ClusterInfo>* info_wrapper = dynamic_cast<WrapperData<ClusterInfo>*>(response[1]);
    if (info_wrapper == nullptr) {
      st_promise.set_value(Status::ArgumentError("ConnectToServers: Response should be ClusterInfo"));
      return;
    }
    info = info_wrapper->Internal();
    st_promise.set_value(Status::Ok());
  }));
  std::future<Status> st_future = st_promise.get_future();
  st_future.wait();
  Status st = st_future.get();
  PS_CHECK_STATUS(st);
  if (info.GetServers().size() == 0) {
    return Status::Unknown("No servers found");
  }
  offset_.push_back(1);
  for (auto item : info.server_size_) {
    offset_.push_back(offset_.back() + item);
  }
  for (auto&& item : info.GetServers()) {
    if (!client_lib_->Connect(offset_[item.GetServerType()] + item.GetId(), item.GetIp() + ":" + std::to_string(item.GetPort()))) {
      return Status::NetworkError(
          "Server[" + std::to_string(item.GetServerType()) + "][" + std::to_string(item.GetId()) + "] Connect Failed "
          + item.GetIp() + ":" + std::to_string(item.GetPort()));
    }
  }
  int error = client_lib_->CheckServer();
  if (error != 0) {
    for (auto&& item : info.GetServers()) {
      if (offset_[item.GetServerType()] + item.GetId() == (size_t)error - 1) {
        return Status::NetworkError(
            "Server[" + std::to_string(item.GetId()) + "] Connect Failed "
            + item.GetIp() + ":" + std::to_string(item.GetPort()));
      }
    }
  }
  return Status::Ok();
}

int ClientWrapperImpl::ServerSize(int id) {
  return offset_[id + 1] - offset_[id];
}

int ClientWrapperImpl::ServerTypeSize() {
  return offset_.size() - 1;
}

} //namespace client
} //namespace ps

