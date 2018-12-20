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

#ifndef XDL_CORE_FRAMEWORK_DEVICE_H_
#define XDL_CORE_FRAMEWORK_DEVICE_H_

#include <string>

#include "xdl/core/lib/status.h"
#include "xdl/core/lib/thread_pool.h"
#include "xdl/core/framework/graph_def.h"
#include "xdl/core/framework/allocator.h"

namespace xdl {

class OpKernelBase;
class OpKernelContext;

class Device {
 public:
  Device(Allocator* allocator) : allocator_(allocator) {}
  virtual ~Device() {}
  virtual std::string DeviceType() = 0;
  virtual void ScheduleToRun(ThreadPool* tp, OpKernelBase* op,
                             OpKernelContext* ctx);

  void* Allocate(size_t size) {
    return allocator_->Allocate(size);
  }

  void Deallocate(void* buf) {
    return allocator_->Deallocate(buf);
  }

  Allocator* GetAllocator() {
    return allocator_.get();
  }

 private:
  RefCountedPtr<Allocator> allocator_;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_DEVICE_H_

