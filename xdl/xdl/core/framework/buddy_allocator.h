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

#ifndef XDL_CORE_FRAMEWORK_BUDDY_ALLOCATOR_H_
#define XDL_CORE_FRAMEWORK_BUDDY_ALLOCATOR_H_

#include <vector>
#include <mutex>

#include "xdl/core/framework/allocator.h"

namespace xdl {

class BuddyAllocator : public Allocator {
 public:
  BuddyAllocator(Allocator* internal, size_t allocate_size, size_t block_size);
  ~BuddyAllocator();
  void* Allocate(size_t size) override;
  void Deallocate(void* buf) override;
  bool TryDeallocate(void* buf);
 private:
  struct Chunk {
    std::unique_ptr<uint32_t[]> masks;
    std::unique_ptr<bool[]> allocate_flag;
    void* ptr;
  };

  void* FindBlock(int id, uint32_t size);
  void AllocateNewChunk();
  int FindChunk(void* buf);
  void Free(int id, void* buf);

  std::mutex mu_;
  RefCountedPtr<Allocator> internal_;
  size_t allocate_size_;
  size_t block_size_;
  uint32_t chunk_size_;
  std::vector<Chunk> chunks_;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_BUDDY_ALLOCATOR_H_

