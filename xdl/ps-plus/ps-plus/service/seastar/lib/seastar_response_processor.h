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

#ifndef PS_SERVICE_SEASTAR_LIB_SEASTAR_RESPONSE_PROCESSOR_H_
#define PS_SERVICE_SEASTAR_LIB_SEASTAR_RESPONSE_PROCESSOR_H_

#include <core/ps_queue_hub/queue_work_item.hh>
#include <core/ps_queue_hub/queue_hub.hh>
#include <core/ps_coding/message_processor.hh>
#include <core/ps_coding/fragment.hh>
#include <service/client.hh>

#include "ps-plus/common/serializer.h"

#include "common.h"
#include "ps-plus/common/status.h"
#include "closure_manager.h"
#include "callback_closure.h"

namespace ps {
class Data;

namespace network {
class session_context;
}

namespace service {
namespace seastar {
class SeastarResponseProcessor : public ps::coding::MessageProcessor {
 public:
  ~SeastarResponseProcessor() override {}

  bool ProcessError(ps::network::SessionContext* sc) {
    if (sc->GetIsBadStatus()) {
      const std::vector<uint64_t>& sequences = sc->GetSavedSequenceWhenBroken();
      for (auto& id: sequences) {
        Closure* cs = ps::service::seastar::ClosureManager::GetClosure(id);
        if (cs == NULL) {
          std::cerr << "miss callback for packet[" << id << "]" << std::endl;
          continue;
        }

        CallBackClosure* cb = dynamic_cast<CallBackClosure*>(cs);
        cb->SetStatus(SeastarStatus::NetworkError());
        cb->Run();
      }

      return true;
    }

    return false;
  }

  bool ProcessTimeout(ps::network::SessionContext* sc) {
    if (sc->GetHaveTimeOutPacket()) {
      const std::vector<uint64_t>& sequences = sc->GetSavedSequenceWhenBroken();
      for (auto& id: sequences) {
        Closure* cs = ps::service::seastar::ClosureManager::GetClosure(id);
        if (cs == NULL) {
          std::cerr << "miss callback for packet[" << id << "]" << std::endl;
          continue;
        }

        CallBackClosure* cb = dynamic_cast<CallBackClosure*>(cs);
        cb->SetStatus(SeastarStatus::Timeout());
        cb->Run();
      }

      return true;
    }

    return false;
  }

  ::seastar::future<> Process(ps::network::SessionContext* sc) override {
    if (ProcessError(sc) || ProcessTimeout(sc)) {
      return ::seastar::make_ready_future<>();
    }

    Closure* cs = ps::service::seastar::ClosureManager::GetClosure(this->GetMessageHeader().mSequence);
    if (cs == NULL) {
      std::cerr << "miss callback for packet[" << this->GetMessageHeader().mSequence << "]" << std::endl;
      return ::seastar::make_ready_future<>();
    }

    CallBackClosure* cb = dynamic_cast<CallBackClosure*>(cs);
    SeastarStatus status;
    ParseStatus(&status);
    if (status == SeastarStatus::Ok()) {
      std::vector<size_t> serialize_ids;
      ParseSerializeIDS(&serialize_ids);
      ps::serializer::Fragment buf;
      buf.base = const_cast<char*>(GetDataBuffer().get());
      buf.size = GetDataBuffer().size();
      std::vector<ps::Data*> response_datas;
      response_datas.reserve(serialize_ids.size());        
      size_t offset = 0;
      ps::serializer::MemGuard mem_guard;
      for (size_t& id: serialize_ids) {
        ps::Data* data = nullptr;
        size_t len;
        if (!ps::serializer::DeserializeAny<ps::Data>(id, &buf, offset, &data, &len, mem_guard).IsOk()) {
          status = SeastarStatus::ClientDeserializeFailed();
          break;
        }

        offset += len;
        response_datas.push_back(data);
      }

      cb->SetResponseData(response_datas);
      cb->SetMemGuard(mem_guard);
    }

    cb->SetStatus(status);
    cb->Run();
    return ::seastar::make_ready_future<>();
  }

  void ParseStatus(SeastarStatus* status) {
    int32_t* begin = reinterpret_cast<int32_t*>(
        const_cast<char*>(GetMetaBuffer().begin()));    
    *status = SeastarStatus((SeastarStatus::ErrorCode)(*begin));
  }

  void ParseSerializeIDS(std::vector<size_t>* ids) {
    size_t* begin = reinterpret_cast<size_t*>(
        const_cast<char*>(GetMetaBuffer().begin()) + 
        sizeof(int32_t));
    size_t len = GetMetaBuffer().size() / sizeof(size_t);
    ids->assign(begin, begin + len);
  }
};

PS_DECLARE_MESSAGE_PROCESSOR_CLASS_ID(SeastarResponseProcessor, SEASTAR_RESPONSE_PROCESSOR_ID);

} // namespace seastar
} // namespace service
} // namespace ps

#endif //PS_SERVICE_SEASTAR_LIB_SEASTAR_RESPONSE_PROCESSOR_H_
