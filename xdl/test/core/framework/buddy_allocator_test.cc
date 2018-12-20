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

#include "gtest/gtest.h"
#include "xdl/core/framework/buddy_allocator.h"

#include <deque>

using xdl::Allocator;
using xdl::BuddyAllocator;

namespace {
class MockAllocator : public Allocator {
 public:
  static constexpr int BlockSize = 1 << 20;
  static constexpr char* BlockOffset = reinterpret_cast<char*>(42);
  MockAllocator() : counter_(0) {}
  static void* Buffer(uint64_t counter) { return BlockOffset + counter * BlockSize; }
  void* Allocate(size_t size) override {
    return Buffer(counter_++);
  }
  void Deallocate(void* b) override {
    counter_--;
  }
  int Counter() {
    return counter_;
  }
 private:
  int counter_;
};

class BuddyAllocatorSimulator {
 public:
  BuddyAllocatorSimulator(int depth) : depth_(depth) {}
  struct AllocateBlock {
    int block_depth;
    int chunk_id;
    int block_id;
  };
  void Allocate(AllocateBlock* block) {
    for (size_t i = 0; i < chunks_.size(); i++) {
      if (FindBlock(i, block)) {
        return;
      }
    }
    NewChunk();
    FindBlock(chunks_.size() - 1, block);
  }
  void Deallocate(AllocateBlock* block) {
    int k = ((1 << block->block_depth) - 1) + block->block_id / (1 << (depth_ - block->block_depth));
    Set(block->chunk_id, k, true);
  }
  bool FindBlock(int chunk_id, AllocateBlock* block) {
    Chunk& chunk = chunks_[chunk_id];
    int beg = (1 << block->block_depth) - 1, size = 1 << block->block_depth;
    for (int i = 0; i < size; i++) {
      if (chunk.free[beg + i]) {
        block->chunk_id = chunk_id;
        block->block_id = i * (1 << (depth_ - block->block_depth));
        Set(chunk_id, beg + i, false);
        return true;
      }
    }
    return false;
  }
  void NewChunk() {
    chunks_.emplace_back();
    Chunk& chunk = chunks_.back();
    chunk.free = std::vector<bool>((1 << (depth_ + 1)) - 1, true);
  }
 private:
  struct Chunk {
    std::vector<bool> free;
  };

  void Set(int chunk_id, int offset, bool real) {
    Chunk& chunk = chunks_[chunk_id];
    int beg = offset, end = offset + 1;
    for (int beg = offset, end = offset + 1;
         beg < (1 << (depth_ + 1)) - 1;
         beg = beg * 2 + 1, end = end * 2 + 1) {
      for (int j = beg; j < end; j++) {
        chunk.free[j] = real;
      }
    }
    int k = offset;
    while (k > 0) {
      k = (k - 1) / 2;
      chunk.free[k] = chunk.free[k * 2 + 1] && chunk.free[k * 2 + 2];
    }
  }

  std::vector<Chunk> chunks_;
  int depth_;
};

}

TEST(TestBuddyAllocator, SmallSlice1) {
  std::mt19937 rnd(42);
  MockAllocator* mock_allocator = new MockAllocator;
  BuddyAllocator* buddy_allocator = new BuddyAllocator(mock_allocator, 2048, 8);
  for (int i = 0; i < 256; i++) {
    void* buffer = MockAllocator::Buffer(0);
    void* pre_ptr = static_cast<char*>(buffer) + 8 * i;
    void* allocate_ptr = buddy_allocator->Allocate(size_t(rnd()) % 8 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(1, mock_allocator->Counter());
  for (int i = 0; i < 256; i++) {
    void* buffer = MockAllocator::Buffer(1);
    void* pre_ptr = static_cast<char*>(buffer) + 8 * i;
    void* allocate_ptr = buddy_allocator->Allocate(size_t(rnd()) % 8 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(2, mock_allocator->Counter());
  buddy_allocator->UnRef();
  ASSERT_EQ(0, mock_allocator->Counter());
  mock_allocator->UnRef();
}

TEST(TestBuddyAllocator, SmallSlice2) {
  std::mt19937 rnd(42);
  MockAllocator* mock_allocator = new MockAllocator;
  BuddyAllocator* buddy_allocator = new BuddyAllocator(mock_allocator, 2048, 8);
  for (int i = 0; i < 128; i++) {
    void* buffer = MockAllocator::Buffer(0);
    void* pre_ptr = static_cast<char*>(buffer) + 16 * i;
    void* allocate_ptr = buddy_allocator->Allocate(size_t(rnd()) % 8 + 8 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(1, mock_allocator->Counter());
  for (int i = 0; i < 128; i++) {
    void* buffer = MockAllocator::Buffer(1);
    void* pre_ptr = static_cast<char*>(buffer) + 16 * i;
    void* allocate_ptr = buddy_allocator->Allocate(size_t(rnd()) % 8 + 8 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(2, mock_allocator->Counter());
  buddy_allocator->UnRef();
  ASSERT_EQ(0, mock_allocator->Counter());
  mock_allocator->UnRef();
}

TEST(TestBuddyAllocator, SimulatorUniform) {
  int block_size = 8, depth = 8;
  for (int i = 0; i <= 15; i++) {
    std::mt19937 rnd(i);
    MockAllocator* mock_allocator = new MockAllocator;
    BuddyAllocator* buddy_allocator = new BuddyAllocator(mock_allocator, block_size << depth, block_size);
    BuddyAllocatorSimulator simulator(depth);
    std::vector<BuddyAllocatorSimulator::AllocateBlock> blocks;
    for (int j = 0; j < 1000; j++) {
      bool allocate;
      if (blocks.empty()) {
        allocate = true;
      } else {
        allocate = (size_t)rnd() % 100 < (100 - i * 5);
      }
      if (allocate) {
        blocks.emplace_back();
        BuddyAllocatorSimulator::AllocateBlock& block = blocks.back();
        int allocate_size = (size_t)rnd() % (block_size << depth) + 1;
        int allocate_depth;
        for (allocate_depth = 0; allocate_depth <= depth; allocate_depth++) {
          if (allocate_size <= (block_size << allocate_depth)) {
            break;
          }
        }
        block.block_depth = depth - allocate_depth;
        void* allocate_ptr = buddy_allocator->Allocate(allocate_size);
        simulator.Allocate(&block);
        void* pre_ptr = static_cast<char*>(MockAllocator::Buffer(block.chunk_id))
                        + 8 * block.block_id;
        ASSERT_EQ(pre_ptr, allocate_ptr);
      } else {
        size_t id = (size_t)rnd() % blocks.size();
        BuddyAllocatorSimulator::AllocateBlock block = blocks[id];
        blocks[id] = blocks.back();
        blocks.pop_back();
        void* pre_ptr = static_cast<char*>(MockAllocator::Buffer(block.chunk_id))
                        + 8 * block.block_id;
        simulator.Deallocate(&block);
        buddy_allocator->Deallocate(pre_ptr);
      }
    }
    buddy_allocator->UnRef();
    mock_allocator->UnRef();
  }
}

TEST(TestBuddyAllocator, SimulatorDepthUniform) {
  int block_size = 8, depth = 8;
  for (int i = 0; i <= 15; i++) {
    std::mt19937 rnd(i);
    MockAllocator* mock_allocator = new MockAllocator;
    BuddyAllocator* buddy_allocator = new BuddyAllocator(mock_allocator, block_size << depth, block_size);
    BuddyAllocatorSimulator simulator(depth);
    std::vector<BuddyAllocatorSimulator::AllocateBlock> blocks;
    for (int j = 0; j < 1000; j++) {
      bool allocate;
      if (blocks.empty()) {
        allocate = true;
      } else {
        allocate = (size_t)rnd() % 100 < (100 - i * 5);
      }
      if (allocate) {
        blocks.emplace_back();
        BuddyAllocatorSimulator::AllocateBlock& block = blocks.back();
        int allocate_size;
        int allocate_depth = (size_t)rnd() % (depth + 1);
        if (allocate_depth == 0) {
          allocate_size = (size_t)rnd() % block_size + 1;
        } else {
          allocate_size = (size_t)rnd() % (block_size << (allocate_depth - 1))
                          + (block_size << (allocate_depth - 1)) + 1;
        }
        block.block_depth = depth - allocate_depth;
        void* allocate_ptr = buddy_allocator->Allocate(allocate_size);
        simulator.Allocate(&block);
        void* pre_ptr = static_cast<char*>(MockAllocator::Buffer(block.chunk_id))
                        + 8 * block.block_id;
        ASSERT_EQ(pre_ptr, allocate_ptr);
      } else {
        size_t id = (size_t)rnd() % blocks.size();
        BuddyAllocatorSimulator::AllocateBlock block = blocks[id];
        blocks[id] = blocks.back();
        blocks.pop_back();
        void* pre_ptr = static_cast<char*>(MockAllocator::Buffer(block.chunk_id))
                        + 8 * block.block_id;
        simulator.Deallocate(&block);
        buddy_allocator->Deallocate(pre_ptr);
      }
    }
    buddy_allocator->UnRef();
    mock_allocator->UnRef();
  }
}
