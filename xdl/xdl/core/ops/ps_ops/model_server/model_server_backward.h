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

#ifndef XDL_CORE_OPS_PS_OPS_MODEL_SERVER_MODEL_SERVER_BACKWARD_H_
#define XDL_CORE_OPS_PS_OPS_MODEL_SERVER_MODEL_SERVER_BACKWARD_H_

#include "xdl/core/lib/status.h"
#include "xdl/core/lib/singleton.h"
#include "xdl/core/lib/concurrent_queue.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"

namespace xdl {

struct PsModelServerBackwardItem {
  ps::Tensor ids;
  ps::Tensor grads;
  std::function<void(Status)> done;
  bool run;
};

class PsModelServerBackwardQueue : public Singleton<PsModelServerBackwardQueue> {
 public:
  int NewHandle() {
    queues_.emplace_back(new ConcurrentQueue<PsModelServerBackwardItem*>);
    return queues_.size() - 1;
  }

  ConcurrentQueue<PsModelServerBackwardItem*>* Queue(int handle) {
    return queues_[handle].get();
  }

  void Wait(int handle, std::function<void()> waiter) {
    queues_[handle]->Wait(waiter);
  }

  void Pop(int handle, std::function<void(PsModelServerBackwardItem*)> cb) {
    queues_[handle]->Pop(cb);
  }

  void Push(int handle, PsModelServerBackwardItem* item) {
    queues_[handle]->Push(item);
  }

  void Run(int handle, ps::Tensor ids, ps::Tensor grads, std::function<void(ps::Status)> done) {
    PsModelServerBackwardItem* item = new PsModelServerBackwardItem;
    item->ids = ids;
    item->grads = grads;
    item->done = [done](Status status) {
      done(XDL2PS::ConvertStatus(status));
    };
    Push(handle, item);
  }
 private:
  std::vector<std::unique_ptr<ConcurrentQueue<PsModelServerBackwardItem*>>> queues_;
};

} // namespace xdl

#endif  // XDL_CORE_OPS_PS_OPS_MODEL_SERVER_MODEL_SERVER_BACKWARD_H_

