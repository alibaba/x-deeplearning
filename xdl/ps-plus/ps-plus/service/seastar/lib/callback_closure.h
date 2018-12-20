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

#ifndef PS_SERVICE_SEASTAR_LIB_CALLBACK_CLOSURE_H_
#define PS_SERVICE_SEASTAR_LIB_CALLBACK_CLOSURE_H_

#include <functional>
#include <core/ps_common.hh>

#include "ps-plus/common/data.h"
#include "ps-plus/common/serializer.h"

#include "common.h"
#include "seastar_status.h"

namespace ps {
namespace service {
namespace seastar {

class CallBackClosure : public Closure {
 public:
  using CallBack = std::function<void(const SeastarStatus&, const std::vector<ps::Data*>& response)>;
  CallBackClosure(const CallBack& cb) :
    cb_(cb) {
  }
    
  CallBackClosure(CallBack&& cb) :
    cb_(std::move(cb)) {
  }    

  ~CallBackClosure() override {
    for (Data* data: response_datas_) {
      delete data;
    }
  }

  virtual void Run() {
    cb_(status_, response_datas_);
    delete this;
  }    

  void SetResponseData(std::vector<ps::Data*>& datas) {
    std::swap(response_datas_, datas);
  }

  void SetMemGuard(const ps::serializer::MemGuard& mem_guard) {
    mem_guard_ = mem_guard;
  }

  void SetStatus(const SeastarStatus& status) {
    status_ = status;
  }

 private:
  std::function<void(const SeastarStatus&, const std::vector<ps::Data*>&)> cb_;    
  std::vector<ps::Data*> response_datas_;
  ps::serializer::MemGuard mem_guard_;
  SeastarStatus status_;
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif //PS_SERVICE_SEASTAR_LIB_CALLBACK_CLOSURE_H_
