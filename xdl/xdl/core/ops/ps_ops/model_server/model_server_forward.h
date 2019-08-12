/* Copyright 2018 Alibaba Group. All Rights Reserved.

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

#ifndef XDL_CORE_OPS_PS_OPS_MODEL_SERVER_MODEL_SERVER_FORWARD_H_
#define XDL_CORE_OPS_PS_OPS_MODEL_SERVER_MODEL_SERVER_FORWARD_H_

#include "xdl/core/lib/status.h"
#include "xdl/core/lib/singleton.h"
#include "xdl/core/lib/concurrent_queue.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"

namespace xdl {

struct PsModelServerForwardItem {
  ps::Tensor ids;
  std::function<void(ps::Tensor)> ok;
  std::function<void(Status)> err;
  bool run;
};

class PsModelServerForwardQueue : public Singleton<PsModelServerForwardQueue> {
 public:
  int NewHandle() {
    queues_.emplace_back(new ConcurrentQueue<PsModelServerForwardItem*>);
    return queues_.size() - 1;
  }

  ConcurrentQueue<PsModelServerForwardItem*>* Queue(int handle) {
    return queues_[handle].get();
  }

  void Wait(int handle, std::function<void()> waiter) {
    queues_[handle]->Wait(waiter);
  }

  void Pop(int handle, std::function<void(PsModelServerForwardItem*)> cb) {
    queues_[handle]->Pop(cb);
  }

  void Push(int handle, PsModelServerForwardItem* item) {
    queues_[handle]->Push(item);
  }

  void Run(int handle, ps::Tensor ids, std::function<void(ps::Status, ps::Tensor)> done) {
    PsModelServerForwardItem* item = new PsModelServerForwardItem;
    item->ids = ids;
    item->ok = [done](ps::Tensor tensor) {
      done(ps::Status::Ok(), tensor);
    };
    item->err = [done](Status status) {
      done(XDL2PS::ConvertStatus(status), ps::Tensor());
    };
    Push(handle, item);
  }
 private:
  std::vector<std::unique_ptr<ConcurrentQueue<PsModelServerForwardItem*>>> queues_;
};

} // namespace xdl

#endif  // XDL_CORE_OPS_PS_OPS_MODEL_SERVER_MODEL_SERVER_FORWARD_H_

