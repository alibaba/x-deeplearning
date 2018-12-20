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

#ifndef XDL_CORE_FRAMEWORK_SLAB_ALLOCATOR_H_
#define XDL_CORE_FRAMEWORK_SLAB_ALLOCATOR_H_

#include <vector>
#include <mutex>
#include <memory>

#include "xdl/core/framework/allocator.h"

namespace xdl {

class SlabAllocator : public Allocator {
 public:
  SlabAllocator(Allocator* internal, size_t allocate_size, size_t segment_size,
                size_t block_size);
  ~SlabAllocator();
  void* Allocate(size_t size) override;
  void Deallocate(void* buf) override;
  bool TryDeallocate(void* buf);
  static constexpr int kMaxBlockForAllocate = 4;
  static constexpr int kChunkSize = 64;
  static constexpr int kAllocationSize = 1024;
 private:
  struct LinkNode {
    LinkNode* prev;
    LinkNode* next;
    int32_t line_id;
    int32_t chunk_id;
    uint64_t allocate_mask;
    uint64_t allocate_head_mask;
  };
  struct SegmentLine {
    std::unique_ptr<LinkNode[]> link_list;
    void* ptr;
  };
  struct Segment {
    std::mutex mu_;
    std::vector<SegmentLine> lines;
    std::vector<LinkNode> links;
    Segment() : links(kMaxBlockForAllocate + 1) {
      for (int i = 0; i < kMaxBlockForAllocate + 1; i++) {
        links[i].prev = &links[i];
        links[i].next = &links[i];
      }
    }
  };

  void* SegmentAllocate(int seg_id, int32_t block);
  void AllocateSegmentLine(int seg_id);
  bool FindBlock(void* ptr, int* seg_id, int* line_id, int* block_id);
  void Free(int seg_id, int line_id, int block_id);

  std::mutex allocate_mu_;
  size_t allocations_size_;
  std::unique_ptr<void*[]> allocations_;
  std::unique_ptr<Segment[]> segment_;
  std::atomic<int32_t> counter_;
  RefCountedPtr<Allocator> internal_;
  size_t allocate_size_;
  size_t segment_size_;
  size_t block_size_;
  size_t chunk_count_;
  size_t segment_allocate_size_;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_SLAB_ALLOCATOR_H_

