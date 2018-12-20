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

#ifndef PS_COMMON_HASHMAP_H
#define PS_COMMON_HASHMAP_H

#include <vector>
#include <atomic>
#include <deque>
#include <mutex>
#include <random>
#include "rd_lock.h" 
#include "memblock.h"

namespace ps {

#define ITEM_SIZE 2

class HashMap {
 public:

  HashMap(int64_t hint);
  ~HashMap();
  void SetMaxCache(int max_cache) {
    max_cache_ = max_cache;    
  }

  int Get(int64_t* keys, int64_t size, int64_t jump,
          std::vector<int64_t>* ids, std::vector<int64_t>* reused_ids);

  int GetWithAddProbability(int64_t* keys, int64_t size, int64_t jump, double add_probability,
          std::vector<int64_t>* ids, std::vector<int64_t>* reused_ids);
    
  int GetWithoutAdd(int64_t* keys, int64_t size, int64_t jump, std::vector<int64_t>* ids);

  int Del(int64_t* keys, int64_t size, int64_t jump); 

  struct HashMapStruct {
    std::vector<HashMapItem> items;
    size_t counter;
  };
  int GetHashKeys(HashMapStruct* result);
  // Only be used in clear hashmap
  int SetHashKeys(const HashMapStruct& result);
  static const int64_t NOT_ADD_ID;
 private:
  inline int64_t GetHashKey(int64_t x, int64_t y) {
    x = ((x & 0xAAAAAAAAAAAAAAAAL) >> 1) + ((x & 0x5555555555555555L) << 1);
    y = ((y & 0xFFFFFFFF00000000L) >> 32) + ((y & 0x00000000FFFFFFFFL) << 32);
    int64_t h = x ^ y;
    return h == 0 ? -1 : h;
  }

  inline SMemBlock<ITEM_SIZE>* GetHashBlockList(int ver, int64_t x, int64_t y) {
    int64_t pos = GetHashKey(x, y) % nodes_vec_[ver].size();
    pos = pos >= 0 ? pos : pos + nodes_vec_[ver].size();
    return &(nodes_vec_[ver][pos]);
  }

  inline int64_t TryGetOne(int ver, int64_t x, int64_t y);
  inline int64_t AddOne(int ver, int64_t x, int64_t y, int64_t id, bool* id_reused);
  inline void DelOne(int ver, int64_t x, int64_t y, int64_t id = -1);
  inline int64_t GetId(bool* id_reused); 
  int ReplayLogs(int ver);
  void ResetVer(int ver);
  // expand space when there is not enough space to fill data
  int ExpandSpace(int ver);
  static int64_t GetHashSize(int64_t hint, bool scale = false);

  static const int VERSIONS = 2; 
  static const int MAX_CACHED = 10000;
  static const int64_t INVALID_ID = -1;
  static std::default_random_engine dre;
  static std::uniform_real_distribution<double> urd;

  using HashNode = std::vector<SMemBlock<ITEM_SIZE>>;

  enum OperateType {
    ADD_OP = 0,
    DEL_OP
  };

  struct LogNode {
    LogNode() {
    }

    LogNode(int64_t x, int64_t y, int64_t id, OperateType type) {
      this->x = x;
      this->y = y;
      this->id = id;
      this->type = type;
    }

    int64_t x;
    int64_t y;
    int64_t id;
    OperateType type;
  };

  int64_t counter_;
  int64_t hint_;
  int update_counter_;
  int max_cache_;
  std::atomic<int> current_ver_;
  ReadWriteLock rwlock_[VERSIONS];
  std::mutex write_lock_;
  // hash table node
  HashNode nodes_vec_[VERSIONS];
  MemBlockPool<ITEM_SIZE>* data_pool_vec_[VERSIONS]; 
  std::deque<int64_t> reused_ids_list_;

  // log for adding or removing  
  std::vector<LogNode> logs_[VERSIONS];  
};

int64_t HashMap::TryGetOne(int ver, int64_t x, int64_t y) {
  SMemBlock<ITEM_SIZE>* cur_block = GetHashBlockList(ver, x, y); 
  HashMapItem* item;
  while (cur_block) {
    item = cur_block->Get(x, y);
    if (nullptr != item) {
      // found
      return item->id;  
    }
    cur_block = cur_block->Next();
  } 
  return INVALID_ID;
}

int64_t HashMap::GetId(bool* id_reused) {
  // first, check reused_ids_list_
  *id_reused = false;
  if (!reused_ids_list_.empty()) {
    int64_t id = reused_ids_list_.front();
    reused_ids_list_.pop_front();
    *id_reused = true;
    return id;
  }
  // second, produce new id
  return counter_++; 
}

int64_t HashMap::AddOne(int ver, int64_t x, int64_t y, int64_t id,
    bool* id_reused) {
  *id_reused = false;
  int64_t old_id = TryGetOne(ver, x, y);
  if (INVALID_ID != old_id) {
    return old_id;
  }
  int64_t new_id;
  if (id < 0) {
    new_id = GetId(id_reused);
  } else {
    new_id = id;
  }
  
  SMemBlock<ITEM_SIZE>* cur_block = GetHashBlockList(ver, x, y);
  int ret = 0;
  while (cur_block) {
    ret = cur_block->Push(x, y, new_id); 
    if (0 == ret) {
      if (id < 0) {
        // will need replay log
        logs_[ver].emplace_back(x, y, new_id, ADD_OP);
      }
      return new_id;
    }
    if (nullptr == cur_block->Next()) {
      SMemBlock<ITEM_SIZE>* tmp = data_pool_vec_[ver]->Borrow();
      if (nullptr == tmp) {
        // need expand space
        if (0 != ExpandSpace(ver)) {
          return INVALID_ID;
        }
        // NOTE: should get hash block list again
        SMemBlock<ITEM_SIZE>* new_head = GetHashBlockList(ver, x, y);
        if (nullptr == new_head) {
          return INVALID_ID;
        }
        cur_block = new_head;
      } else {
        cur_block->Next() = tmp;
        cur_block = cur_block->Next();
      }
    } else {
      cur_block = cur_block->Next();
    }
  }
  return INVALID_ID;
}

void HashMap::DelOne(int ver, int64_t x, int64_t y, int64_t id) {
  SMemBlock<ITEM_SIZE>* head = GetHashBlockList(ver, x, y);
  SMemBlock<ITEM_SIZE>* cur_block = head;
  int64_t old_id;
  while (cur_block) {
    if (0 == cur_block->Del(x, y, &old_id)) {
      if (-1 == id) {
        if (old_id >= 0) {
          // release id
          reused_ids_list_.push_back(old_id);
          logs_[ver].emplace_back(x, y, old_id, DEL_OP);
        } else {
          // do nothing, because it has been logical deleted
        }
      }
      break;
    }  
    cur_block = cur_block->Next(); 
  }
  // return MemBlock to pool if it's empty
  SMemBlock<ITEM_SIZE>* tmp = head;
  while (tmp && tmp->Next()) {
    if (tmp->Next()->Empty()) {
      SMemBlock<ITEM_SIZE>* next = tmp->Next()->Next();
      data_pool_vec_[ver]->Return(tmp->Next());
      tmp->Next() = next; 
    }
    tmp = tmp->Next();
  }

  // do logical deletion to read buffer
  int read_ver = current_ver_.load();
  SMemBlock<ITEM_SIZE>* read_cur_block = GetHashBlockList(read_ver, x, y);
  while (read_cur_block) {
    HashMapItem* item = read_cur_block->Get(x, y);
    if (nullptr != item) {
      item->id = INVALID_ID;
      break;
    }
    read_cur_block = read_cur_block->Next();
  } 
}

} //ps

#endif //PS_COMMON_HASHMAP_H
