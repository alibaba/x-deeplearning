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

#include "xdl/core/framework/slab_allocator.h"
#include "xdl/core/utils/logging.h"
#include <iostream>

namespace xdl {

namespace {

int32_t level(uint64_t v) {
  uint64_t v2 = v & (v >> 1);
  if (v2 != 0) {
    if ((v2 & (v2 >> 2)) != 0) {
      return 4;
    } else if ((v2 & (v >> 2)) != 0) {
      return 3;
    } else {
      return 2;
    }
  } else {
    if (v != 0) {
      return 1;
    } else {
      return 0;
    }
  }
}

}  // namespace

SlabAllocator::SlabAllocator(Allocator* internal, size_t allocate_size,
                             size_t segment_size, size_t block_size)
    : counter_(0), internal_(internal), allocate_size_(allocate_size),
      segment_size_(segment_size), block_size_(block_size) {
  XDL_CHECK(allocate_size_ >= (segment_size_ * block_size_ * kChunkSize))
    << "AllocateSize should be more than SegmentSize * BlockSize * ChunkSize";
  XDL_CHECK(allocate_size_ % (segment_size_ * block_size_ * kChunkSize) == 0)
    << "AllocateSize should be diveded by SegmentSize * BlockSize * ChunkSize";
  chunk_count_ = allocate_size_ / segment_size_ / block_size_ / kChunkSize;
  segment_allocate_size_ = allocate_size_ / segment_size_;
  segment_.reset(new Segment[segment_size_]);
  allocations_.reset(new void*[kAllocationSize]);
  for (int i = 0; i < kAllocationSize; i++) {
    allocations_[i] = nullptr;
  }
  allocations_size_ = 0;
}

SlabAllocator::~SlabAllocator() {
  for (size_t i = 0; i < allocations_size_; i++) {
    internal_->Deallocate(allocations_[i]);
  }
}

void* SlabAllocator::Allocate(size_t size) {
  if (size == 0) {
    return nullptr;
  }
  int32_t block = size / block_size_;
  if (size % block_size_ != 0) {
    block++;
  }
  XDL_CHECK(block <= kMaxBlockForAllocate) << "Cannot Allocate size " << size;
  int32_t seg_id = counter_++ % segment_size_;
  return SegmentAllocate(seg_id, block);
}

void SlabAllocator::Deallocate(void* buf) {
  XDL_CHECK(TryDeallocate(buf)) << "Buddy Allocator Error! Invalid buf " << buf;
}

bool SlabAllocator::TryDeallocate(void* buf) {
  if (buf == nullptr) {
    return true;
  }
  int seg_id, line_id, block_id;
  bool ok = FindBlock(buf, &seg_id, &line_id, &block_id);
  if (!ok) {
    return false;
  }
  Free(seg_id, line_id, block_id);
  return true;
}

void* SlabAllocator::SegmentAllocate(int seg_id, int32_t block) {
  Segment& segment = segment_[seg_id];
  std::unique_lock<std::mutex> lock(segment.mu_);
  int i = seg_id;
  for (i = block; i <= kMaxBlockForAllocate; i++) {
    if (segment.links[i].next != &segment.links[i]) {
      break;
    }
  }
  LinkNode* node;
  if (i == kMaxBlockForAllocate + 1) {
    AllocateSegmentLine(seg_id);
    node = segment.links[kMaxBlockForAllocate].next;
  } else {
    node = segment.links[i].next;
  }
  node->prev->next = node->next;
  node->next->prev = node->prev;
  uint64_t mask = (1ul << block) - 1;
  int loc;
  for (loc = 0; loc < kChunkSize; loc++) {
    if (((node->allocate_mask >> loc) & mask) == mask) {
      break;
    }
  }
  XDL_CHECK(loc < kChunkSize) << "Internal Error";
  if (node->chunk_id == 1) {
    LinkNode* nodex = node - 1;
  }
  node->allocate_mask = node->allocate_mask & ~(mask << loc);
  node->allocate_head_mask = node->allocate_head_mask | (1ul << loc);
  int32_t node_level = level(node->allocate_mask);
  node->prev = &segment.links[node_level];
  node->next = segment.links[node_level].next;
  node->prev->next = node;
  node->next->prev = node;
  return static_cast<char*>(segment.lines[node->line_id].ptr)
    + block_size_ * (kChunkSize * node->chunk_id + loc);
}
void SlabAllocator::AllocateSegmentLine(int seg_id) {
  void* ptr; {
    std::unique_lock<std::mutex> lock(allocate_mu_);
    if (segment_[seg_id].lines.size() < allocations_size_) {
      ptr = static_cast<char*>(allocations_[segment_[seg_id].lines.size()]) +
            segment_allocate_size_ * seg_id;
    } else {
      allocations_[allocations_size_] = internal_->Allocate(allocate_size_);
      ptr = static_cast<char*>(allocations_[allocations_size_]) +
            segment_allocate_size_ * seg_id;
      allocations_size_++;
    }
  }
  int32_t line_id = segment_[seg_id].lines.size();
  segment_[seg_id].lines.emplace_back();
  SegmentLine& line = segment_[seg_id].lines.back();
  LinkNode* prev = &segment_[seg_id].links[kMaxBlockForAllocate];
  LinkNode* end = prev->next;
  line.ptr = ptr;
  line.link_list.reset(new LinkNode[chunk_count_]);
  for (size_t i = 0; i < chunk_count_; i++) {
    LinkNode* node = &line.link_list[i];
    node->prev = prev;
    prev->next = node;
    prev = node;
    node->line_id = line_id;
    node->chunk_id = i;
    //                     0123456789ABCDEF    F*16
    node->allocate_mask = 0xFFFFFFFFFFFFFFFFL;
    node->allocate_head_mask = 0;
  }
  line.link_list[chunk_count_ - 1].next = end;
  end->prev = &line.link_list[chunk_count_ - 1];
}

bool SlabAllocator::FindBlock(
    void* ptr, int* seg_id, int* line_id, int* block_id) {
  size_t line_size = allocations_size_;
  size_t i;
  for (i = 0; i < line_size; i++) {
    void* beg = allocations_[i];
    void* end = static_cast<char*>(allocations_[i]) + allocate_size_;
    if (ptr >= beg && ptr < end) {
      int diff = static_cast<char*>(ptr) - static_cast<char*>(beg);
      *line_id = i;
      *seg_id = diff / segment_allocate_size_;
      *block_id = (diff % segment_allocate_size_) / block_size_;
      return true;
    }
  }
  return false;
}

void SlabAllocator::Free(int seg_id, int line_id, int block_id) {
  Segment& segment = segment_[seg_id];
  std::unique_lock<std::mutex> lock(segment.mu_);
  int chunk_id = block_id / kChunkSize;
  int b_id = block_id % kChunkSize;
  LinkNode* node = &segment.lines[line_id].link_list[chunk_id];

  node->prev->next = node->next;
  node->next->prev = node->prev;

  node->allocate_mask = node->allocate_mask | (1ul << b_id);
  node->allocate_head_mask = node->allocate_head_mask & ~(1ul << b_id);
  uint64_t mask = (1ul << b_id) << 1;
  while ((node->allocate_mask & mask) == 0 &&
         (node->allocate_head_mask & mask) == 0 && mask != 0) {
    node->allocate_mask = node->allocate_mask | mask;
    mask = mask << 1;
  }
  int32_t node_level = level(node->allocate_mask);
  node->prev = &segment.links[node_level];
  node->next = segment.links[node_level].next;
  node->prev->next = node;
  node->next->prev = node;
  
}

}  // namespace xdl
