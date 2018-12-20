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

#ifndef PS_SERVICE_SEASTAR_LIB_SEASTAR_RESPONSE_SERIALIZER_H_
#define PS_SERVICE_SEASTAR_LIB_SEASTAR_RESPONSE_SERIALIZER_H_

#include <core/ps_coding/message_serializer.hh>
#include <core/ps_coding/message_processor.hh>

#include "ps-plus/common/data.h"
#include "ps-plus/common/serializer.h"

#include "common.h"
#include "seastar_status.h"

namespace ps {
namespace service {
namespace seastar {

class SeastarResponseSerializer : public ps::coding::MessageSerializer {
 public:
  SeastarResponseSerializer(ps::coding::MessageProcessor* processor,
                            std::vector<ps::Data*>& request_datas,
                            const ps::serializer::MemGuard& mem_guard,
                            const SeastarStatus& status)
    : processor_(processor)
    , mem_guard_(mem_guard)
    , status_(status) {
    request_datas_.swap(request_datas);
  }

  ~SeastarResponseSerializer() override {
    for (ps::Data* data: request_datas_) {
      delete data;
    }

    for (ps::Data* data: response_datas_) {
      delete data;
    }
  }

  void Serialize() override {
    // std::cout << "serialize seq:" << GetSequence() << std::endl;    
    SetProcessorClassId(SEASTAR_RESPONSE_PROCESSOR_ID);        
    std::vector<ps::serializer::Fragment> fragments;
    std::vector<size_t> ids;
    fragments.reserve(response_datas_.size());
    ids.reserve(response_datas_.size());
    for (ps::Data* data: response_datas_) {
      size_t id;
      if (!ps::serializer::SerializeAny<ps::Data>(data, &id, &fragments, 
                                                  mem_guard_).IsOk()) {
        status_ = SeastarStatus::ServerSerializeFailed();
        break;
      }

      ids.push_back(id);
    }

    for (auto& item: fragments) {
      AppendFragment((ps::coding::Fragment&)item);
    }

    AddStatus(status_);
    AddSerializeIDs(ids);
  }

  void AddStatus(SeastarStatus& status) {
    AppendMeta(status.Data(), status.Size());
  }

  void AddSerializeIDs(const std::vector<size_t>& ids) {
    for (size_t id: ids) {
      AppendMeta(id);
    }
  }

  std::vector<ps::Data*>* MutableResponseData() {
    return &response_datas_;
  }  

  const std::vector<ps::Data*>& RequestData() const {
    return request_datas_;
  }  

 private:
  std::vector<ps::Data*> response_datas_;
  std::vector<ps::Data*> request_datas_;
  ps::coding::MessageProcessor* processor_;
  ps::serializer::MemGuard mem_guard_;
  SeastarStatus status_;
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif //PS_SERVICE_SEASTAR_LIB_SEASTAR_RESPONSE_SERIALIZER_H_
