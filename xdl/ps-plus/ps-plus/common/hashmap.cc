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

#include <iostream>
#include <algorithm>
#include <unordered_set>
#include "hashmap.h"

namespace ps {

int64_t qmul(int64_t a, int64_t n, int64_t m) {
  int64_t ans = 0;
  while (n > 0) {
    if(n % 2 == 1) ans = (ans + a) % m;
    a = (a + a) % m;
    n /= 2;
  }
  return ans;
}

int64_t qpow(int64_t a, int64_t n, int64_t m) {
  int64_t ans = 1;
  while (n > 0) {
    if(n % 2 == 1) ans = qmul(ans, a, m);
    a = qmul(a, a, m);
    n /= 2;
  }
  return ans;
}

bool Miller_Rabbin(int64_t n, int64_t a) {
  int64_t r = 0, s = n - 1, j;
  if(n % a == 0) {
    return false;
  }
  while(s % 2 == 1) {
    s /= 2;
    r++;
  }
  int64_t k = qpow(a, s, n);
  if (k == 1) {
    return true;
  }
  for (j = 0; j < r; j++, k = qmul(k, k, n)) {
    if (k == n - 1) {
      return true;
    }
  }
  return false;
}

bool IsPrime(int n) {
  static const int tab[] = {2, 3, 5, 7};
  for(int i = 0;i < 4; i++) {
    if(n == tab[i])
      return true;
    if(!Miller_Rabbin(n, tab[i]))
      return false;
  }
  return true;
}

HashMap::HashMap(int64_t hint) :
    counter_(0),
    hint_(hint),
    update_counter_(0),
    max_cache_(MAX_CACHED),
    current_ver_(0) {
  int64_t actual_size = GetHashSize(hint);
  for (int i = 0; i < VERSIONS; i++) {
    nodes_vec_[i].resize(actual_size);
    data_pool_vec_[i] = new MemBlockPool<ITEM_SIZE>(
        actual_size * 0.5);
  }
}

HashMap::~HashMap() {
  for (int i = 0; i < VERSIONS; i++) {
    nodes_vec_[i].clear();
    if (data_pool_vec_[i]) {
      delete data_pool_vec_[i];
    }
  }
}

int64_t HashMap::GetHashSize(int64_t hint, bool scale) {
  int64_t x = hint;
  if (scale) {
    x = hint * 2 + 1;
  } else {
    if ((x & 1) == 0) {
      x += 1;
    } 
  } 
  while (!IsPrime(x)) {x += 2;}
  return x;
}

int HashMap::ExpandSpace(int ver) {
  int64_t new_size = GetHashSize(nodes_vec_[ver].size() * 2);
  HashNode tmp_nodes(new_size); 
  int pool_size = data_pool_vec_[ver]->Size();
  MemBlockPool<ITEM_SIZE>* new_data_pool =
    new MemBlockPool<ITEM_SIZE>(pool_size * 2);
  // swap the space between origin space and new one 
  tmp_nodes.swap(nodes_vec_[ver]);
  int64_t tmp_size = tmp_nodes.size();
  MemBlockPool<ITEM_SIZE>* tmp_data_pool = data_pool_vec_[ver]; 
  data_pool_vec_[ver] = new_data_pool;

  // do rehash
  bool id_reused;
  std::cerr<<"ExpandSpace, old_size: "<<tmp_size<<" new size:"<<new_size<<" ver:"<<ver<<std::endl;
  for (int i = 0; i < tmp_size; i++) {
    auto cur_block = &(tmp_nodes[i]);
    while (cur_block) {
      HashMapItem* item_array = cur_block->Array();
      if (nullptr == item_array) {
        return -1;
      }
      for (int i = 0; i < ITEM_SIZE; ++i) {
        if (item_array[i].id < 0) {
          continue;
        }
        int64_t id = AddOne(ver, item_array[i].x, item_array[i].y, item_array[i].id, &id_reused);
        if (INVALID_ID == id) {
          return -1; 
        }
      }  
      cur_block = cur_block->Next(); 
    }
  }

  // release old space 
  tmp_nodes.clear(); 
  delete tmp_data_pool;
  return 0;
}

int HashMap::ReplayLogs(int ver) {
  bool id_reused;
  for (int i = 0; i < VERSIONS; i++) {
    if (i != ver) {
      for (const LogNode& node : logs_[i]) {
        if (ADD_OP == node.type) {
          if (INVALID_ID == AddOne(ver, node.x, node.y, node.id, &id_reused)) {
            return -1; 
          } 
        } else if (DEL_OP == node.type) {
          DelOne(ver, node.x, node.y, node.id);
        }
      }
    }
  }
  return 0;
}

void HashMap::ResetVer(int ver) {
  logs_[(ver + 1) % VERSIONS].clear();
  current_ver_.store(ver);
  update_counter_ = 0;
  ver = (current_ver_.load() + 1) % VERSIONS;
  rwlock_[ver].wrlock();
  ReplayLogs(ver);
  rwlock_[ver].unlock();
}

int HashMap::Get(int64_t* keys, int64_t size, int64_t jump,
          std::vector<int64_t>* ids, std::vector<int64_t>* reused_ids) {
    return GetWithAddProbability(keys, size, jump, 1, ids, reused_ids);
}

int HashMap::GetWithAddProbability(int64_t* keys, int64_t size, int64_t jump, double add_probability,
          std::vector<int64_t>* ids, std::vector<int64_t>* reused_ids) {
  int ver = current_ver_.load();
  ids->resize(size);
  reused_ids->clear();
  int64_t insert_list = -1;
  rwlock_[ver].rdlock();
  for (int64_t i = 0; i < size; i++) {
    int64_t x = keys[i * jump];
    int64_t y = keys[i * jump + 1];
    int64_t rst = TryGetOne(ver, x, y);
    if (INVALID_ID == rst) {
        (*ids)[i] = insert_list;
        insert_list = i;
    } else {
        (*ids)[i] = rst;
    }
  }
  rwlock_[ver].unlock();
  if (insert_list != -1) {
    write_lock_.lock();
    ver = (current_ver_.load() + 1) % VERSIONS;
    rwlock_[ver].wrlock();
    bool id_reused;
    while (insert_list != -1) {
      int64_t x = keys[insert_list * jump];
      int64_t y = keys[insert_list * jump + 1];
      int64_t next = (*ids)[insert_list];
      //need to search this ver too to make sure x,y not in hash list
      int64_t rst = TryGetOne(ver, x, y);
      if (rst < 0) {
          if (urd(dre) <= add_probability) {
              int64_t new_id = AddOne(ver, x, y, -1, &id_reused);
              if (new_id < 0) {
                  return -1; 
              }
              (*ids)[insert_list] = new_id;
              if (id_reused) {
                  reused_ids->push_back(new_id);
              }
          } else {
              (*ids)[insert_list] = NOT_ADD_ID;
          }
      } else {
          (*ids)[insert_list] = rst;
      }
      insert_list = next;
      update_counter_++;
    }
    rwlock_[ver].unlock();
    if (update_counter_ > max_cache_) {
      ResetVer(ver);
    }  
    write_lock_.unlock();
  }
  return 0;
}

int HashMap::GetWithoutAdd(int64_t* keys, int64_t size, int64_t jump, std::vector<int64_t>* ids) {
  write_lock_.lock();
  int ver = (current_ver_.load() + 1) % VERSIONS;
  ResetVer(ver);
  write_lock_.unlock();

  ids->resize(size);
  rwlock_[ver].rdlock();
  for (int64_t i = 0; i < size; i++) {
    int64_t x = keys[i * jump];
    int64_t y = keys[i * jump + 1];
    int64_t rst = TryGetOne(ver, x, y);
    if (INVALID_ID == rst) {
        (*ids)[i] = -1;
    } else {
        (*ids)[i] = rst;
    }
  }
  rwlock_[ver].unlock();
  return 0;
}

int HashMap::Del(int64_t* keys, int64_t size, int64_t jump) {
  write_lock_.lock();
  int ver = (current_ver_.load() + 1) % VERSIONS;
  rwlock_[ver].wrlock();
  for (int64_t i = 0; i < size; i++) {
    int64_t x = keys[i * jump];
    int64_t y = keys[i * jump + 1];
    DelOne(ver, x, y);  
    update_counter_++;
  }
  rwlock_[ver].unlock();
  if (update_counter_ > max_cache_) {
    ResetVer(ver);
  }  
  write_lock_.unlock();
  return 0;
}

int HashMap::GetHashKeys(HashMap::HashMapStruct* data) {
  write_lock_.lock();
  int ver = (current_ver_.load() + 1) % VERSIONS;
  ResetVer(ver);
  data->counter = counter_;
  write_lock_.unlock();

  ver = current_ver_.load();
  rwlock_[ver].rdlock();
  for (auto&& item : nodes_vec_[ver]) {
    for (auto iter = & item; iter != nullptr; iter = iter->Next()) {
      auto items = iter->Array();
      for (size_t i = 0; i < ITEM_SIZE; i++) {
        if (items[i].id != INVALID_ID) {
          data->items.push_back(items[i]);
        }
      }
    }
  }
  rwlock_[ver].unlock();
  return 0;
}

int HashMap::SetHashKeys(const HashMap::HashMapStruct& data) {
  std::unordered_set<size_t> ids;
  for (auto item : data.items) {
    ids.insert(item.id);
    for (int ver = 0; ver < VERSIONS; ver++) {
      SMemBlock<ITEM_SIZE>* cur_block = GetHashBlockList(ver, item.x, item.y); 
      while (cur_block->Next()) {
        cur_block = cur_block->Next();
      }
      bool add_new_node = true;
      for (int i = 0; i < ITEM_SIZE; i++) {
        if (cur_block->Array()[i].id == INVALID_ID) {
          cur_block->Array()[i] = item;
          add_new_node = false;
          break;
        }
      }
      if (add_new_node) {
        SMemBlock<ITEM_SIZE>* tmp = data_pool_vec_[ver]->Borrow();
        if (tmp == nullptr) {
          return -1;
        }
        cur_block->Next() = tmp;
        tmp->Array()[0] = item;
      }
    }
  }
  for (size_t i = 0; i < data.counter; i++) {
    if (ids.find(i) == ids.end()) {
      reused_ids_list_.push_back(i);
    }
  }
  counter_ = data.counter;
  return 0;
}

std::default_random_engine HashMap::dre;
std::uniform_real_distribution<double> HashMap::urd(0.0, 1.0);
const int64_t HashMap::NOT_ADD_ID = -2;
} //ps
