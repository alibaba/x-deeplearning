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

#include "xdl/core/framework/buddy_allocator.h"

#include "xdl/core/utils/logging.h"
#include <iostream>

namespace xdl {

namespace {

inline bool is_pow_of_2(uint32_t x) {
  return !(x & (x-1));
}

inline bool is_pow_of_2(size_t x) {
  return !(x & (x-1));
}

inline uint32_t next_pow_of_2(uint32_t x) {
  if (is_pow_of_2(x)) {
    return x;
  }
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

inline size_t next_pow_of_2(size_t x) {
  if (is_pow_of_2(x)) {
    return x;
  }
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

}

BuddyAllocator::BuddyAllocator(
    Allocator* internal, size_t allocate_size, size_t block_size)
  : internal_(internal), allocate_size_(allocate_size),
    block_size_(block_size), chunk_size_(allocate_size / block_size) {
  XDL_CHECK(is_pow_of_2(allocate_size))
    << "Buddy Allocator allocate size shoud be pow of 2";
  XDL_CHECK(is_pow_of_2(block_size))
    << "Buddy Allocator block size shoud be pow of 2";
  XDL_CHECK(allocate_size > block_size)
    << "Buddy Allocator allocate size shoud greater than block size";
  XDL_CHECK(block_size > 0)
    << "Buddy Allocator block size should greater than 0";
}

BuddyAllocator::~BuddyAllocator() {
  for (auto& chunk : chunks_) {
    internal_->Deallocate(chunk.ptr);
  }
}

void* BuddyAllocator::Allocate(size_t size) {
  if (size == 0) {
    return nullptr;
  }
  std::unique_lock<std::mutex> lock(mu_);
  XDL_CHECK(size <= allocate_size_) << "size should less than allocate_size";
  size_t s = next_pow_of_2(size) / block_size_;
  s = s == 0 ? 1 : s;
  for (size_t i = 0; i < chunks_.size(); i++) {
    void* buf = FindBlock(i, s);
    if (buf != nullptr) {
      return buf;
    }
  }
  AllocateNewChunk();
  return FindBlock(chunks_.size() - 1, s);
}

void BuddyAllocator::Deallocate(void* buf) {
  XDL_CHECK(TryDeallocate(buf)) << "Buddy Allocator Error! Invalid buf " << buf;
}

bool BuddyAllocator::TryDeallocate(void* buf) {
  if (buf == nullptr) {
    return true;
  }
  std::unique_lock<std::mutex> lock(mu_);
  int chunk_id = FindChunk(buf);
  if (chunk_id == -1) {
    return false;
  }
  Free(chunk_id, buf);
  return true;
}

void* BuddyAllocator::FindBlock(int id, uint32_t size) {
  uint32_t mask = size;
  Chunk& chunk = chunks_[id];
  uint32_t* masks = chunk.masks.get();
  if ((masks[0] & mask) == 0) {
    return nullptr;
  }
  uint32_t cur_offset = 0;
  uint32_t goal_offset = chunk_size_ / size - 1;
  while (cur_offset < goal_offset) {
    if ((masks[cur_offset * 2 + 1] & mask) != 0) {
      cur_offset = cur_offset * 2 + 1;
    } else {
      cur_offset = cur_offset * 2 + 2;
    }
  }
  void* ret = static_cast<void*>(
      static_cast<char*>(chunk.ptr) +
      static_cast<size_t>(cur_offset - goal_offset) * size * block_size_);
  masks[cur_offset] = 0;
  chunk.allocate_flag[cur_offset] = true;
  while (cur_offset > 0) {
    cur_offset = (cur_offset - 1) / 2;
    masks[cur_offset] = masks[cur_offset * 2 + 1] | masks[cur_offset * 2 + 2];
  }
  return ret;
}

void BuddyAllocator::AllocateNewChunk() {
  void* buf = internal_->Allocate(allocate_size_);
  chunks_.emplace_back();
  Chunk& chunk = chunks_.back();
  chunk.ptr = buf;
  chunk.masks.reset(new uint32_t[chunk_size_ * 2]);
  chunk.allocate_flag.reset(new bool[chunk_size_ * 2]);
  uint32_t* masks = chunk.masks.get();
  bool* allocate_flag = chunk.allocate_flag.get();
  for (uint32_t mask = chunk_size_; mask != 0; mask = mask / 2) {
    uint32_t offset = chunk_size_ / mask - 1;
    uint32_t real_mask = mask * 2 - 1;
    for (uint32_t i = offset; i < offset * 2 + 1; i++) {
      masks[i] = real_mask;
    }
  }
  for (uint32_t i = 0; i < chunk_size_ * 2; i++) {
    allocate_flag[i] = false;
  }
}

int BuddyAllocator::FindChunk(void* buf) {
  for (size_t i = 0; i < chunks_.size(); i++) {
    auto ptr_diff =
        static_cast<char*>(buf) - static_cast<char*>(chunks_[i].ptr);
    if (ptr_diff >= 0 && ptr_diff < allocate_size_) {
      return i;
    }
  }
  return -1;
}

void BuddyAllocator::Free(int id, void* buf) {
  Chunk& chunk = chunks_[id];
  bool* allocate_flag = chunk.allocate_flag.get();
  uint32_t* masks = chunk.masks.get();
  auto ptr_diff =
      static_cast<char*>(buf) - static_cast<char*>(chunk.ptr);
  uint32_t offset = chunk_size_ - 1 + ptr_diff / block_size_;
  uint32_t vmask = 1;
  bool updated = false;
  while (true) {
    if (updated) {
      uint32_t lhs = masks[offset * 2 + 1];
      uint32_t rhs = masks[offset * 2 + 2];
      if (lhs == vmask - 1 && rhs == vmask - 1) {
        masks[offset] = vmask * 2 - 1;
      } else {
        masks[offset] = lhs | rhs;
      }
    } else {
      if (allocate_flag[offset]) {
        allocate_flag[offset] = false;
        masks[offset] = vmask * 2 - 1;
        updated = true;
      }
    }
    if (offset == 0) {
      break;
    }
    vmask = vmask * 2;
    offset = (offset - 1) / 2;
  }
}

}  // namespace xdl
