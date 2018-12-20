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

#ifndef PS_SERVICE_SEASTAR_LIB_SEASTAR_REQUEST_ITEM_H_
#define PS_SERVICE_SEASTAR_LIB_SEASTAR_REQUEST_ITEM_H_

#include <core/ps_queue_hub/queue_work_item.hh>

#include "seastar_request_serializer.h"

namespace ps {
class Data;

namespace network {
class SessionContext;
}

namespace coding {
class MessageSerializer;
}

namespace service {
namespace seastar {

class SeastarRequestItem : public ps::network::SeastarWorkItem {
 public:
  SeastarRequestItem(int32_t server_id,
                     int32_t thread_id, 
                     int32_t func_id,
                     ps::network::NetworkContext* nc,
                     const std::vector<ps::Data*>& request_datas,
                     bool delete_request_data) :
    ps::network::SeastarWorkItem(nc, server_id, false), 
    thread_id_(thread_id),
    serializer_(new SeastarRequestSerializer(
                    func_id, 
                    request_datas, 
                    delete_request_data)) {
  }

  ~SeastarRequestItem() override {}

  ::seastar::future<> Run() override {
    serializer_->SetUserThreadId(thread_id_);
    serializer_->SetSequence(this->GetSequence());
    return GetSessionContext()->Write(serializer_);
  }

  int GetThreadID() const { 
    return thread_id_; 
  }

 private:
  int thread_id_;
  ps::coding::MessageSerializer* serializer_;
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif // PS_SERVICE_SEASTAR_LIB_SEASTAR_REQUEST_ITEM_H_
