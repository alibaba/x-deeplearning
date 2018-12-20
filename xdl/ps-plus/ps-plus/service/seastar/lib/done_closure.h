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

#ifndef PS_SERVICE_SEASTAR_LIB_DONE_CLOSURE_H_
#define PS_SERVICE_SEASTAR_LIB_DONE_CLOSURE_H_

#include <functional>
#include <thread>
#include <mutex>

#include <core/ps_queue_hub/queue_hub.hh>
#include <core/reactor.hh>
#include <core/ps_common.hh>

#include "common.h"
#include "seastar_response_item.h"

namespace ps {
namespace network {
class SessionContext;
}

namespace coding {
class MessageSerializer;
}

namespace service {
namespace seastar {

class DoneClosure : public Closure {
 public:
  DoneClosure(ps::network::SessionContext* sc, 
              ps::coding::MessageSerializer* serializer,
			  bool should_enqueue = false) 
    : sc_(sc)
    , serializer_(serializer)
    , thread_id_(std::this_thread::get_id())
    , cpu_id_(::seastar::engine().cpu_id())
    , should_enqueue_(should_enqueue) {
  } 

  ~DoneClosure() override {}

  virtual void Run() {
    if (std::this_thread::get_id() != thread_id_) {
      std::unique_lock<std::mutex> lock(global_mu_);
      std::pair<ps::network::QueueHub<ps::network::Item>*, 
                ps::network::QueueHub<ps::network::Item>*> qhr = 
        ps::network::QueueHubFactory::GetInstance().GetDefaultQueueForServer();
//        ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>("SEASTAR");
      ps::network::QueueHub<ps::network::Item>* outputHub = qhr.second;
      SeastarResponseItem* item = new SeastarResponseItem(sc_, serializer_);
      outputHub->Enqueue(item, cpu_id_, 0, NULL);
      ps::network::Future<ps::network::Item> f = outputHub->GetFuture(cpu_id_, 0);
      f.Get();
    } else {
      // server write via queue that can promise the packet seq in multi thread mode
      if (should_enqueue_) {
        ::seastar::engine().GetWriteQueuePoller().Enqueue(sc_, serializer_);
      } else {
        sc_->Write(serializer_).then([] () {});
      }
    }
    
    delete this;
  }    

 private:
  static std::mutex global_mu_;
  ps::network::SessionContext* sc_;
  ps::coding::MessageSerializer* serializer_;
  std::thread::id thread_id_;
  size_t cpu_id_;
  bool should_enqueue_;
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif //PS_SERVICE_SEASTAR_LIB_DONE_CLOSURE_H_
