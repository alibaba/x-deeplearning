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

#ifndef XDL_CORE_FRAMEWORK_ALLOCATOR_H_
#define XDL_CORE_FRAMEWORK_ALLOCATOR_H_

#include <unordered_map>
#include <mutex>

#include "xdl/core/lib/refcount.h"
#include "xdl/core/lib/singleton.h"

namespace xdl {

class Allocator : public RefCounted {
 public:
  virtual void* Allocate(size_t size) = 0;
  virtual void Deallocate(void* buf) = 0;
};

class AllocatorManager : public Singleton<AllocatorManager> {
 public:
  Allocator* Get(const std::string& name,
                 std::function<Allocator*()> creator);
 private:
  std::mutex mu_;
  std::unordered_map<std::string, RefCountedPtr<Allocator>> allocators_;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_ALLOCATOR_H_

