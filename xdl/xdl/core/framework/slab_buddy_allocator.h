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

#ifndef XDL_CORE_FRAMEWORK_SLAB_BUDDY_ALLOCATOR_H_
#define XDL_CORE_FRAMEWORK_SLAB_BUDDY_ALLOCATOR_H_

#include <vector>
#include <mutex>

#include "xdl/core/framework/allocator.h"
#include "xdl/core/framework/slab_allocator.h"
#include "xdl/core/framework/buddy_allocator.h"

namespace xdl {

class SlabBuddyAllocator : public Allocator {
 public:
  SlabBuddyAllocator(Allocator* internal, size_t allocate_size,
                     size_t block_size, size_t slab_segment_size);
  void* Allocate(size_t size) override;
  void Deallocate(void* buf) override;
 private:
  size_t allocate_size_;
  size_t block_size_;
  RefCountedPtr<Allocator> internal_;
  RefCountedPtr<SlabAllocator> slab_allocator_;
  RefCountedPtr<BuddyAllocator> buddy_allocator_;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_SLAB_BUDDY_ALLOCATOR_H_

