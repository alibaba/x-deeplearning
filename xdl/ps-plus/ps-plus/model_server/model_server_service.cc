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

#include "ps-plus/model_server/model_server_service.h"

#include "ps-plus/common/net_utils.h"
#include "ps-plus/common/reliable_kv.h"
#include "ps-plus/message/server_info.h"
#include "ps-plus/service/seastar/lib/callback_closure.h"
#include "ps-plus/service/seastar/lib/done_closure.h"
#include <thread>
#include <tuple>
#include <future>
#include <tuple>
#include "ps-plus/common/logging.h"

namespace ps {
namespace modelserver {

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

ModelServerService::ModelServerService(
    ModelServer* model_server, const std::string& scheduler,
    int server_type, int server_id) {
  model_server_.reset(model_server);
  port_ = NetUtils::GetAvailablePort();
  NetUtils::GetDefaultIP(ip_);
  server_type_ = server_type;
  server_id_ = server_id;
  core_num_ = NetUtils::GetAvailableCpuNum();
  stop_ = false;
  server_version_ = NewRandomVersion();
  scheduler_kv_addr_ = scheduler;
}

ModelServerService::~ModelServerService() {
  stop_ = true;
  if (register_server_loop_ != nullptr) {
    register_server_loop_->join();
  }
  if (seastar_lib_ != nullptr) {
    seastar_lib_->Stop();
  }
}

Status ModelServerService::Init() {
  PS_CHECK_STATUS(model_server_->Init());
  seastar_lib_.reset(new ps::service::seastar::SeastarServerClientLib(port_, core_num_, core_num_, CLIENT_THREAD_NUM));
  seastar_lib_->RegisterServerFunc(func_ids::kModelServerFlush,
      [this](const std::vector<ps::Data*>& inputs, 
             std::vector<ps::Data*>* outputs, 
             ps::service::seastar::DoneClosure* done) {
    Flush(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kModelServerForward,
      [this](const std::vector<ps::Data*>& inputs, 
             std::vector<ps::Data*>* outputs, 
             ps::service::seastar::DoneClosure* done) {
    RequestForward(inputs, outputs, done);
  });
  seastar_lib_->RegisterServerFunc(func_ids::kModelServerBackward,
      [this](const std::vector<ps::Data*>& inputs, 
             std::vector<ps::Data*>* outputs, 
             ps::service::seastar::DoneClosure* done) {
    RequestBackward(inputs, outputs, done);
  });
  seastar_lib_->Start();
  register_server_loop_.reset(new std::thread([this]{RegisterServer();}));
  return Status::Ok();
}

void ModelServerService::Flush(
    const std::vector<Data*>& inputs,
    std::vector<Data*>* outputs,
    ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 1) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("Flush: Need 1 inputs")));
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  if (ver == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("RequestForward: inputs[0] should be Version")));
    done->Run();
    return;
  }
  ver_.store(kUnusedVersion);
  model_server_->Flush(
      [outputs, done](Status st) {
    outputs->push_back(new WrapperData<Status>(st));
    done->Run();
  });
  ver_.store(ver->Internal());
}

void ModelServerService::RequestForward(
    const std::vector<Data*>& inputs,
    std::vector<Data*>* outputs,
    ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 2) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("RequestForward: Need 2 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<Tensor>* ids = dynamic_cast<WrapperData<Tensor>*>(inputs[1]);
  if (ver == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("RequestForward: inputs[0] should be Version")));
    done->Run();
    return;
  }
  if (ids == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("RequestForward: inputs[1] should be Tensor")));
    done->Run();
    return;
  }
  if (ver->Internal() != ver_) {
    outputs->push_back(new WrapperData<Status>(Status::VersionMismatch("RequestForward: Version Mismatch")));
    done->Run();
    return;
  }
  model_server_->RequestForward(ids->Internal(), 
      [outputs, done](Status st, Tensor result) {
    outputs->push_back(new WrapperData<Status>(st));
    if (st.IsOk()) {
      outputs->push_back(new WrapperData<Tensor>(result));
    }
    done->Run();
  });
}

void ModelServerService::RequestBackward(
    const std::vector<Data*>& inputs,
    std::vector<Data*>* outputs,
    ps::service::seastar::DoneClosure* done) {
  if (inputs.size() != 3) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("RequestBackward: Need 3 inputs")));
    done->Run();
    return;
  }
  WrapperData<Version>* ver = dynamic_cast<WrapperData<Version>*>(inputs[0]);
  WrapperData<Tensor>* ids = dynamic_cast<WrapperData<Tensor>*>(inputs[1]);
  WrapperData<Tensor>* grads = dynamic_cast<WrapperData<Tensor>*>(inputs[2]);
  if (ver == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("RequestBackward: inputs[0] should be Version")));
    done->Run();
    return;
  }
  if (ids == nullptr || grads == nullptr) {
    outputs->push_back(new WrapperData<Status>(Status::ArgumentError("RequestBackward: inputs[1,2] should be Tensor")));
    done->Run();
    return;
  }
  if (ver->Internal() != ver_) {
    outputs->push_back(new WrapperData<Status>(Status::VersionMismatch("RequestBackward: Version Mismatch")));
    done->Run();
    return;
  }
  model_server_->RequestBackward(ids->Internal(), grads->Internal(),
      [outputs, done](Status st) {
    outputs->push_back(new WrapperData<Status>(st));
    done->Run();
  });
}

void ModelServerService::RegisterServer() {
  std::string old_scheduler_addr;
  while (!stop_) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::string scheduler_addr;
    Status st = ReliableKV::ReadAny(scheduler_kv_addr_, &scheduler_addr);
    if (!st.IsOk()) {
      LOG(WARNING) << "Cannot Get Scheduler Addr[" << scheduler_kv_addr_ 
                << "]: Status[" << st.ToString() << "]";
      continue;
    }

    size_t pos = scheduler_addr.find('^');
    if (pos == std::string::npos) {
      LOG(WARNING) << "Cannot Get Scheduler Addr[" << scheduler_kv_addr_ 
                << "]: Store[" << scheduler_addr << "]";
    }
    scheduler_addr = scheduler_addr.substr(pos + 1);

    if (scheduler_addr != old_scheduler_addr) {
      seastar_lib_->Connect(0, scheduler_addr);
      old_scheduler_addr = scheduler_addr;
    }

    std::promise<Status> result;
    std::vector<Data*> request_datas = {
      new WrapperData<ps::ServerInfo>(server_type_, server_id_, server_version_, ip_, port_)
    };
    seastar_lib_->Request(0, func_ids::kSchedulerRegisterServer, request_datas,
        new ps::service::seastar::CallBackClosure([&result](const ps::service::seastar::SeastarStatus& sst, 
                                                            const std::vector<ps::Data*>& datas) {
          if (!sst.Success()) {
            LOG(ERROR) << "register server failed, error:" << sst.ToString();
            result.set_value(Status::NetworkError("Scheduler Error"));
          } else {
            LOG(INFO) << "register server result:" << GetNetworkStatus(datas).ToString();
            result.set_value(GetNetworkStatus(datas));
          }
    }));

    result.get_future().wait();
  }
}

}
}
