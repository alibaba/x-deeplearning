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

#ifndef PS_COMMON_MEMBLOCK_H
#define PS_COMMON_MEMBLOCK_H

#include <memory>
#include <vector>

namespace ps {

struct HashMapItem {
  void Clear() {
    x = y = id = -1;
  }

  int64_t x;
  int64_t y;
  int64_t id;
};

template<int item_cnt> 
class SMemBlock {
 public:
  SMemBlock()
      : next_(nullptr) {
    HashMapItem* item = item_array_;
    for (int i = 0; i < item_cnt; ++i, ++item) {
      item->Clear();
    }
  }

  HashMapItem* Array() {
    return item_array_;
  }

  HashMapItem* Get(int64_t x, int64_t y) {
    HashMapItem* item = item_array_;
    for (int i = 0; i < item_cnt; ++i, ++item) {
      if (item->x == x && item->y == y) {
        return item;   
      }
    }
    return nullptr;
  }

  // return:0 OK; !=0 Fail
  int Del(int64_t x, int64_t y, int64_t* id) {
    HashMapItem* item = item_array_;
    for (int i = 0; i < item_cnt; ++i, ++item) {
      if (item->x == x && item->y == y) {
        *id = item->id;
        item->Clear();
        return 0;
      }
    }
    return -1;
  }

  // return:0 OK; !=0 Fail
  int Push(int64_t x, int64_t y, int64_t id) {
    HashMapItem* item = item_array_;
    for (int i = 0; i < item_cnt; ++i, ++item) {
      if (item->id < 0) {
        item->x = x;
        item->y = y;
        item->id = id;
        return 0;
      } 
    }
    return -1;
  }

  bool Empty() {
    HashMapItem* item = item_array_;
    for (int i = 0; i < item_cnt; ++i, ++item) {
      if (item->id >= 0) {
        return false;
      }
    }
    return true;
  } 

  SMemBlock*& Next() {
    return next_; 
  }

 private:
  SMemBlock* next_;
  HashMapItem item_array_[item_cnt];
};

template<int item_cnt>
class MemBlockPool {
 public:
  MemBlockPool(int size) :
      size_(size) {
    if (size < 1) {
      size = 1;
    }
    block_array_.resize(size); 
    for (int i = size - 1; i >= 0; i--) {
      if (i < size - 1) {
          block_array_[i].Next() = &(block_array_[i + 1]); 
      } 
    }
    free_list_ = &(block_array_[0]);   
  }
  
  ~MemBlockPool() {
    block_array_.clear(); 
  }

  int Size() {
    return size_;
  }

  // Borrow MemBlock from pool
  SMemBlock<item_cnt>* Borrow() {
    if (nullptr != free_list_) {
      SMemBlock<item_cnt>* tmp = free_list_;
      free_list_ = free_list_->Next();
      tmp->Next() = nullptr;
      return tmp; 
    }
    return nullptr;
  }

  // Return MemBlock to pool
  void Return(SMemBlock<item_cnt>* mem_block) {
    mem_block->Next() = free_list_;
    free_list_ = mem_block;
  } 

 private:
  int size_;    
  std::vector<SMemBlock<item_cnt>> block_array_;
  SMemBlock<item_cnt>* free_list_;
};

}   // namespace ps

#endif  //PS_COMMON_MEMBLOCK_H
