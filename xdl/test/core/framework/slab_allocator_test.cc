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
#include "xdl/core/framework/slab_allocator.h"

#include <deque>

using xdl::Allocator;
using xdl::SlabAllocator;

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

class SlabAllocatorSimulator {
 public:
  struct AllocateBlock {
    int block_count;
    int line_id;
    int block_id;
  };
  SlabAllocatorSimulator(int segment_size, int chunk_size) {
    segments_.resize(segment_size);
    segment_size_ = segment_size;
    chunk_size_ = chunk_size;
    line_ = std::vector<int>(segment_size, 0);
    segment_ = 0;
  }
  void Allocate(AllocateBlock* block) {
    int seg_id = segment_++ % segment_size_;
    Segment& seg = segments_[seg_id];
    int i;
    for (i = block->block_count; i < 5; i++) {
      if (!seg.chunk_vec[i].empty()) {
        break;
      }
    }
    if (i == 5) {
      NewLine(seg_id);
      i = 4;
    }
    Chunk chunk = seg.chunk_vec[i].front();
    seg.chunk_vec[i].pop_front();
    int s = 0, j = 0;
    for (j = 0; j < 64; j++) {
      if (!chunk.used[j]) {
        s++;
        if (s == block->block_count) {
          break;
        }
      } else {
        s = 0;
      }
    }
    j -= block->block_count;
    j = j + 1;
    block->line_id = chunk.line_id;
    block->block_id = chunk.block_id + j;
    for (int k = 0; k < block->block_count; k++) {
      chunk.used[j + k] = true;
    }
    int free = 0;
    s = 0;
    for (j = 0; j < 64; j++) {
      if (!chunk.used[j]) {
        s++;
        if (s > free) {
          free = s;
        }
      } else {
        s = 0;
      }
    }
    if (free > 4) {
      free = 4;
    }
    seg.chunk_vec[free].push_front(chunk);
  }
  void Deallocate(AllocateBlock* block) {
    Chunk chunk;
    int line_id = block->line_id;
    int block_offset = block->block_id % 64;
    int beg_block_id = block->block_id - block_offset;
    bool found = false;
    for (int i = 0; i < segment_size_; i++) {
      Segment& seg = segments_[i];
      for (int j = 0; j < 5; j++) {
        auto& chunk_vec = seg.chunk_vec[j];
        for (auto iter = chunk_vec.begin(); iter != chunk_vec.end(); iter++) {
          if (iter->line_id == line_id && iter->block_id == beg_block_id) {
            chunk = *iter;
            chunk_vec.erase(iter);
            found = true;
            break;
          }
        }
        if (found) {
          break;
        }
      }
      if (found) {
        for (int i = 0; i < block->block_count; i++) {
          chunk.used[block_offset + i] = false;
        }
        int free = 0;
        int s = 0;
        for (int j = 0; j < 64; j++) {
          if (!chunk.used[j]) {
            s++;
            if (s > free) {
              free = s;
            }
          } else {
            s = 0;
          }
        }
        if (free > 4) {
          free = 4;
        }
        seg.chunk_vec[free].push_front(chunk);
        break;
      }
    }
    if (!found) {
      std::cerr << "internal error" << std::endl;
      abort();
    }
  }
  void NewLine(int seg_id) {
    for (int i = chunk_size_ - 1; i >= 0; i--) {
      Chunk chunk;
      chunk.line_id = line_[seg_id];
      chunk.block_id = 64 * (chunk_size_ * seg_id + i);
      for (int j = 0; j < 64; j++) {
        chunk.used[j] = false;
      }
      segments_[seg_id].chunk_vec[4].push_front(chunk);
    }
    line_[seg_id]++;
  }
 private:
  struct Chunk {
    int line_id;
    int block_id;
    bool used[64];
  };
  struct Segment {
    std::deque<Chunk> chunk_vec[5];
  };
  std::vector<Segment> segments_;
  int segment_size_;
  int chunk_size_;
  std::vector<int> line_;
  int segment_;
};
}

TEST(TestSlabAllocator, SmallSlice1) {
  std::mt19937 rnd(42);
  MockAllocator* mock_allocator = new MockAllocator;
  SlabAllocator* slab_allocator = new SlabAllocator(mock_allocator, 2048, 2, 8);
  for (int i = 0; i < 256; i++) {
    void* buffer = MockAllocator::Buffer(0);
    int seg = i % 2;
    int block_id = i / 2;
    void* pre_ptr = static_cast<char*>(buffer) + 8 * (seg * 128 + block_id);
    void* allocate_ptr = slab_allocator->Allocate(size_t(rnd()) % 8 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(1, mock_allocator->Counter());
  for (int i = 0; i < 256; i++) {
    void* buffer = MockAllocator::Buffer(1);
    int seg = i % 2;
    int block_id = i / 2;
    void* pre_ptr = static_cast<char*>(buffer) + 8 * (seg * 128 + block_id);
    void* allocate_ptr = slab_allocator->Allocate(size_t(rnd()) % 8 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(2, mock_allocator->Counter());
  slab_allocator->UnRef();
  ASSERT_EQ(0, mock_allocator->Counter());
  mock_allocator->UnRef();
}

TEST(TestSlabAllocator, SmallSlice2) {
  std::mt19937 rnd(42);
  MockAllocator* mock_allocator = new MockAllocator;
  SlabAllocator* slab_allocator = new SlabAllocator(mock_allocator, 2048, 2, 8);
  for (int i = 0; i < 128; i++) {
    void* buffer = MockAllocator::Buffer(0);
    int seg = i % 2;
    int block_id = i / 2 * 2;
    void* pre_ptr = static_cast<char*>(buffer) + 8 * (seg * 128 + block_id);
    void* allocate_ptr = slab_allocator->Allocate(size_t(rnd()) % 8 + 8 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(1, mock_allocator->Counter());
  for (int i = 0; i < 128; i++) {
    void* buffer = MockAllocator::Buffer(1);
    int seg = i % 2;
    int block_id = i / 2 * 2;
    void* pre_ptr = static_cast<char*>(buffer) + 8 * (seg * 128 + block_id);
    void* allocate_ptr = slab_allocator->Allocate(size_t(rnd()) % 8 + 8 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(2, mock_allocator->Counter());
  slab_allocator->UnRef();
  ASSERT_EQ(0, mock_allocator->Counter());
  mock_allocator->UnRef();
}

TEST(TestSlabAllocator, SmallSlice3) {
  std::mt19937 rnd(42);
  MockAllocator* mock_allocator = new MockAllocator;
  SlabAllocator* slab_allocator = new SlabAllocator(mock_allocator, 2048, 2, 8);
  for (int i = 0; i < 84; i++) {
    void* buffer = MockAllocator::Buffer(0);
    int seg = i % 2;
    int block_id = i < 42 ? i / 2 * 3 : 64 + (i - 42) / 2 * 3;
    void* pre_ptr = static_cast<char*>(buffer) + 8 * (seg * 128 + block_id);
    void* allocate_ptr = slab_allocator->Allocate(size_t(rnd()) % 8 + 16 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(1, mock_allocator->Counter());
  for (int i = 0; i < 84; i++) {
    void* buffer = MockAllocator::Buffer(1);
    int seg = i % 2;
    int block_id = i < 42 ? i / 2 * 3 : 64 + (i - 42) / 2 * 3;
    void* pre_ptr = static_cast<char*>(buffer) + 8 * (seg * 128 + block_id);
    void* allocate_ptr = slab_allocator->Allocate(size_t(rnd()) % 8 + 16 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(2, mock_allocator->Counter());
  slab_allocator->UnRef();
  ASSERT_EQ(0, mock_allocator->Counter());
  mock_allocator->UnRef();
}

TEST(TestSlabAllocator, SmallSlice4) {
  std::mt19937 rnd(42);
  MockAllocator* mock_allocator = new MockAllocator;
  SlabAllocator* slab_allocator = new SlabAllocator(mock_allocator, 2048, 2, 8);
  for (int i = 0; i < 64; i++) {
    void* buffer = MockAllocator::Buffer(0);
    int seg = i % 2;
    int block_id = i / 2 * 4;
    void* pre_ptr = static_cast<char*>(buffer) + 8 * (seg * 128 + block_id);
    void* allocate_ptr = slab_allocator->Allocate(size_t(rnd()) % 8 + 24 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(1, mock_allocator->Counter());
  for (int i = 0; i < 64; i++) {
    void* buffer = MockAllocator::Buffer(1);
    int seg = i % 2;
    int block_id = i / 2 * 4;
    void* pre_ptr = static_cast<char*>(buffer) + 8 * (seg * 128 + block_id);
    void* allocate_ptr = slab_allocator->Allocate(size_t(rnd()) % 8 + 24 + 1);
    ASSERT_EQ(pre_ptr, allocate_ptr);
  }
  ASSERT_EQ(2, mock_allocator->Counter());
  slab_allocator->UnRef();
  ASSERT_EQ(0, mock_allocator->Counter());
  mock_allocator->UnRef();
}

TEST(TestSlabAllocator, Simulator) {
  int segment_size = 16, chunk_size = 16;
  for (int i = 0; i <= 15; i++) {
    std::mt19937 rnd(i);
    MockAllocator* mock_allocator = new MockAllocator;
    SlabAllocator* slab_allocator = new SlabAllocator(mock_allocator, segment_size * chunk_size * 64 * 8, segment_size, 8);
    SlabAllocatorSimulator simulator(segment_size, chunk_size);
    std::vector<SlabAllocatorSimulator::AllocateBlock> blocks;
    for (int j = 0; j < 10000; j++) {
      bool allocate;
      if (blocks.empty()) {
        allocate = true;
      } else {
        allocate = (size_t)rnd() % 100 < (100 - i * 5);
      }
      if (allocate) {
        blocks.emplace_back();
        SlabAllocatorSimulator::AllocateBlock& block = blocks.back();
        int allocate_size = (size_t)rnd() % 32 + 1;
        block.block_count = (allocate_size - 1) / 8 + 1;
        void* allocate_ptr = slab_allocator->Allocate(allocate_size);
        simulator.Allocate(&block);
        void* pre_ptr = static_cast<char*>(MockAllocator::Buffer(block.line_id))
                        + 8 * block.block_id;
        ASSERT_EQ(pre_ptr, allocate_ptr);
      } else {
        size_t id = (size_t)rnd() % blocks.size();
        SlabAllocatorSimulator::AllocateBlock block = blocks[id];
        blocks[id] = blocks.back();
        blocks.pop_back();
        void* pre_ptr = static_cast<char*>(MockAllocator::Buffer(block.line_id))
                        + 8 * block.block_id;
        simulator.Deallocate(&block);
        slab_allocator->Deallocate(pre_ptr);
      }
    }
    slab_allocator->UnRef();
    mock_allocator->UnRef();
  }
}
