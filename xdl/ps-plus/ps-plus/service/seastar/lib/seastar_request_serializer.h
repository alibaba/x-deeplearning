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

#ifndef PS_SERVICE_SEASTAR_LIB_SEASTAR_REQUEST_SERIALIZER_H_
#define PS_SERVICE_SEASTAR_LIB_SEASTAR_REQUEST_SERIALIZER_H_

#include <functional>
#include <core/ps_queue_hub/queue_work_item.hh>
#include <core/ps_coding/message_serializer.hh>

#include "ps-plus/common/serializer.h"
#include "seastar_status.h"

namespace ps {
namespace service {
namespace seastar {

class SeastarRequestSerializer : public ps::coding::MessageSerializer {
 public:
  SeastarRequestSerializer(int32_t func_id, 
                           const std::vector<Data*>& request_datas,
                           bool delete_request_data)
    : func_id_(func_id)
    , request_datas_(std::move(request_datas))
    , delete_request_data_(delete_request_data) {
  }

  ~SeastarRequestSerializer() override {
    if (delete_request_data_) {
      for (Data* data: request_datas_) {
        delete data;
      }
    }
  }

  void Serialize() override {
    // std::cout << "serialize seq:" << GetSequence() << std::endl;
    SetProcessorClassId(SEASTAR_REQUEST_PROCESSOR_ID);
    AddServerFuncID(func_id_);
    std::vector<ps::serializer::Fragment> fragments;
    std::vector<size_t> ids;
    fragments.reserve(request_datas_.size());
    ids.reserve(request_datas_.size());
    for (Data* data: request_datas_) {
      size_t id;
      ps::Status serialize_st = ps::serializer::SerializeAny<ps::Data>(data, &id, &fragments, mem_guard_);
      if (!serialize_st.IsOk()) {
        std::cerr << serialize_st.ToString() << std::endl;
        st_ = SeastarStatus::ClientSerializeFailed();
        break;
      }

      ids.push_back(id);
    }

    for (auto& item: fragments) {
      AppendFragment((ps::coding::Fragment&)item);
    }

    AddStatus(st_);
    AddSerializeIDs(ids);
  }

  // meta format: func_id | status | serialize_id_1 | serialize_id_2 ...
  void AddServerFuncID(uint64_t server_func_id) {
    AppendMeta(&server_func_id, sizeof(server_func_id));
  }

  void AddStatus(SeastarStatus& st) {
    AppendMeta(st.Data(), st.Size());
  }

  void AddSerializeIDs(const std::vector<size_t>& ids) {
    for (size_t id: ids) {
      AppendMeta(id);
    }
  }
    
 private:
  uint64_t func_id_;
  std::vector<ps::Data*> request_datas_;
  ps::serializer::MemGuard mem_guard_;
  SeastarStatus st_;
  bool delete_request_data_;
};


} // namespace seastar
} // namespace service
} // namespace ps

#endif // PS_SERVICE_SEASTAR_LIB_SEASTAR_REQUEST_SERIALIZER_H_
