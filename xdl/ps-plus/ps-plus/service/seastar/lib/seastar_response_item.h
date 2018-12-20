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

#ifndef PS_SERVICE_SEASTAR_LIB_SEASTAR_RESPONSE_ITEM_H_
#define PS_SERVICE_SEASTAR_LIB_SEASTAR_RESPONSE_ITEM_H_

#include <core/ps_queue_hub/queue_work_item.hh>
#include <core/ps_queue_hub/ps_work_item.hh>
#include <core/ps_coding/message_serializer.hh>
#include <service/session_context.hh>
#include <service/common.hh>

namespace ps {
namespace coding {
class MessageSerializer;
}

namespace service {
namespace seastar {

class SeastarResponseItem : public ps::network::PsWorkItem {
 public:
  SeastarResponseItem(ps::network::SessionContext* sc,
                      ps::coding::MessageSerializer* serializer) 
    : session_context_(sc)
    , serializer_(serializer) {
  }

  void Run() override {
    return;
  }

  ::seastar::future<> Complete() {
    // trick here
    if (!ps::network::ServerConnctionManager::IsAlive(session_context_)) {
      std::cout << "SeastarResponseItem: connection may be closed!" << std::endl;
      return ::seastar::make_ready_future<>();
    }
    return session_context_->Write(serializer_);
  }

 private:
  ps::network::SessionContext* session_context_;
  ps::coding::MessageSerializer* serializer_;
};

} // namespace seastar
} // namespace service
} // ps

#endif // PS_SERVICE_SEASTAR_LIB_SEASTAR_RESPONSE_ITEM_H_
