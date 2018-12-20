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

#include "xdl/core/framework/slab_buddy_allocator.h"

namespace xdl {

SlabBuddyAllocator::SlabBuddyAllocator(
    Allocator* internal, size_t allocate_size,
    size_t block_size, size_t slab_segment_size)
  : allocate_size_(allocate_size), block_size_(block_size),
    internal_(internal),
    slab_allocator_(RefCountedPtr<SlabAllocator>::Create(
          internal, allocate_size, slab_segment_size,
          block_size / SlabAllocator::kMaxBlockForAllocate)),
    buddy_allocator_(RefCountedPtr<BuddyAllocator>::Create(
          internal, allocate_size, block_size)) {
  if (block_size % SlabAllocator::kMaxBlockForAllocate != 0) {
    abort();
  }
  if (block_size > allocate_size) {
    abort();
  }
}

void* SlabBuddyAllocator::Allocate(size_t size) {
  if (size >= allocate_size_) {
    return internal_->Allocate(size);
  } else if (size >= block_size_) {
    return buddy_allocator_->Allocate(size);
  } else {
    return slab_allocator_->Allocate(size);
  }
}

void SlabBuddyAllocator::Deallocate(void* buf) {
  if (slab_allocator_->TryDeallocate(buf)) {
    return;
  }
  if (buddy_allocator_->TryDeallocate(buf)) {
    return;
  }
  internal_->Deallocate(buf);
}

}  // namespace xdl
