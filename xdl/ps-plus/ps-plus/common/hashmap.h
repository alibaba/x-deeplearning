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
#include <iostream>
#include <assert.h>
#include <unordered_set>
#include "rd_lock.h" 
#include "ps-plus/common/qrw_lock.h"
#include "ps-plus/common/bloom_filter.h"
#include "tbb/parallel_for_each.h"
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_vector.h"
#include "ps-plus/common/thread_pool.h"
#include "ps-plus/common/logging.h"

namespace ps {
struct Hash128Key {
  int64_t hash1;
  int64_t hash2;
  bool operator ==(const Hash128Key& rhs) const {
    return hash1 == rhs.hash1 && hash2 == rhs.hash2;
  }
  friend std::ostream& operator<<(std::ostream& os, const Hash128Key& key);
};

template<typename T>
struct HashMapItem {
  T key;
  size_t id;
};

template<typename T>
struct HashMapStruct {
  tbb::concurrent_vector<HashMapItem<T> > items;
  std::atomic<int64_t> count;
};
}

namespace tbb {
template <> struct tbb_hash_compare<ps::Hash128Key> {
  static size_t hash(const ps::Hash128Key& k) {
    return GetHashKey(k.hash1, k.hash2);
  }
  static bool equal(const ps::Hash128Key& k1, const ps::Hash128Key& k2) {
    return k1 == k2;
  }
  static inline int64_t GetHashKey(int64_t x, int64_t y) {
    if (x == y) {
      return x;
    }
    x = ((x & 0xAAAAAAAAAAAAAAAAL) >> 1) + ((x & 0x5555555555555555L) << 1);
    y = ((y & 0xCCCCCCCCCCCCCCCCL) >> 2) + ((y & 0x3333333333333333L) << 2);
    int64_t h = x ^ y;
    return h == 0 ? -1 : h;
  }
};

template <> struct tbb_hash<ps::Hash128Key> {
  size_t operator()(const ps::Hash128Key& k) const {
    return tbb_hash_compare<ps::Hash128Key>::GetHashKey(k.hash1, k.hash2);
  }
};

}

namespace ps {

class HashMap {
 public:  
  HashMap();
  virtual ~HashMap();
  HashMap(const HashMap&) = delete;
  HashMap& operator=(const HashMap&) = delete;
  virtual int64_t Get(const int64_t* keys, size_t size, bool not_insert, float add_probability, std::vector<size_t>* ids, tbb::concurrent_vector<size_t>* reused_ids, size_t* filtered_keys, size_t block_size = 500) = 0;
  virtual void Erase(const int64_t* keys, size_t size) = 0;
  virtual size_t EraseById(const std::string& variable_name, const std::vector<size_t>& ids, tbb::concurrent_vector<size_t>* unfiltered_ids) = 0;
  void SetBloomFilterThrethold(int32_t max_count);  
  static const size_t NOT_ADD_ID;
  static const float FLOAT_EPSILON;
  std::default_random_engine dre;
  std::uniform_real_distribution<float> urd;
  size_t GetSize() {return offset_;}
  virtual size_t GetBucketCount(const std::string& variable_name) = 0;
 protected:
  bool FloatEqual(float v1, float v2);
  std::atomic<size_t> offset_;
  tbb::concurrent_queue<size_t> free_list_;
  int32_t max_count_;
};

template <typename KeyType> class HashMapImpl : public HashMap {
 public:
  HashMapImpl(size_t hint): HashMap(), table_(hint) {}  
  typedef tbb::concurrent_unordered_map<KeyType, size_t, tbb::tbb_hash<KeyType>, std::equal_to<KeyType> > HashTable;
  typedef std::unordered_set<KeyType, tbb::tbb_hash<KeyType>, std::equal_to<KeyType> > NonCocurrentHashTable;
  virtual int64_t Get(const int64_t* keys, size_t size, bool not_insert, float add_probability, std::vector<size_t>* ids, tbb::concurrent_vector<size_t>* reused_ids, size_t* filtered_keys, size_t block_size = 500) {
    ids->resize(size);
    std::atomic<size_t> total_filtered_count(0);
    MultiThreadDo(size, [&](const Range& r) {
          size_t filtered_count = 0;
          for (size_t i = r.begin; i < r.end; i++) {
            KeyType key;
            GetKey(keys, i, &key);
            auto iter = table_.find(key);
            if (iter != table_.end()) {
              (*ids)[i] = iter->second;
              //only not_insert is false(pull request), we use add_probability or bloom filter;
            } else if (!not_insert) {
              if ((FloatEqual(add_probability, 1.0) || urd(dre) <= add_probability)
                && (max_count_ == 0 || GlobalBloomFilter::Instance()->InsertedLookup(&key, sizeof(key), max_count_))
                && (black_list_ == nullptr || black_list_->find(key) == black_list_->end())
                && (white_list_ == nullptr || white_list_->find(key) != white_list_->end())) {
                auto insert = table_.insert(std::make_pair(key, 0));
                if (insert.second) {
                  size_t id;
                  if (free_list_.try_pop(id)) {
                    (*ids)[i] = id;
                    insert.first->second = (*ids)[i];
                    reused_ids->push_back(id);
                  } else {
                    (*ids)[i] = offset_++;
                    insert.first->second = (*ids)[i];
                  }
                } else {
                  (*ids)[i] = insert.first->second;
                }
              } else {
                filtered_count++;
                (*ids)[i] = NOT_ADD_ID;
              }
            }
          }
          total_filtered_count += filtered_count;
          return Status::Ok();
        }, block_size);
    *filtered_keys = total_filtered_count.load();
    return offset_.load();
  }

  virtual void Erase(const int64_t* keys, size_t size) {
    for (size_t i = 0; i < size; i++) {
      KeyType key;
      GetKey(keys, i, &key);
      auto iter = table_.find(key);
      if (iter != table_.end()) {
        size_t wait_erase = iter->second;
        if (table_.unsafe_erase(key)) {
          free_list_.push(wait_erase);
        }
      }
    }
  }

  virtual size_t GetBucketCount(const std::string& variable_name) {
    /*
    LOG_INFO("%s, load_factor %f, max_load_factor %f", variable_name.c_str(), table_.load_factor(), table_.max_load_factor());
    size_t bucket_count = table_.unsafe_bucket_count();
    std::map<size_t, size_t> count_map;
    bool first = true;
    for (size_t i = 0; i < bucket_count; i++) {
      size_t size = table_.unsafe_bucket_size(i);
      if (size != 0 && first) {
        auto iter = table_.unsafe_begin(i);
        first = false;
        while (iter != table_.unsafe_end(i)) {
          std::cout << iter->first << std::endl;
          ++iter;
        }
      }
      if (count_map.find(size) == count_map.end()) {
        count_map[size] = 1;
      } else {
        count_map[size] = count_map[size]+1;
      }
    }
    size_t total = 0;
    for (auto& iter : count_map) {
      total += iter.first * iter.second;
      LOG_INFO("%s, %ld, %ld", variable_name.c_str(), iter.first, iter.second);
    }
    LOG_INFO("%s, total is %ld", variable_name.c_str(), total);
    */
    return table_.unsafe_bucket_count();
  }

  inline void GetItems(HashMapStruct<KeyType>* result) {
    tbb::parallel_for_each(begin(table_), end(table_), [=](const std::pair<KeyType, size_t>& pr) {
          result->items.push_back(HashMapItem<KeyType>{.key=pr.first, .id=pr.second});
        });
    result->count = result->items.size();
    return;
  }

  virtual size_t EraseById(const std::string& variable_name, const std::vector<size_t>& ids, tbb::concurrent_vector<size_t>* unfiltered_ids) {
    std::atomic<size_t> size(0);
    tbb::concurrent_vector<KeyType> keys;
    tbb::parallel_for_each(begin(table_), end(table_), [&](const std::pair<KeyType, size_t>& pr) {
      auto iter = std::lower_bound(ids.begin(), ids.end(), pr.second);
      if (iter != ids.end() && *iter == pr.second) {
        keys.push_back(pr.first);
        free_list_.push(pr.second);
        size++;
      } else {
        unfiltered_ids->push_back(pr.second);
      }
    });
    for (auto&& key : keys) {
      table_.unsafe_erase(key);
    }
    LOG(INFO) << "Filter for " << variable_name << ", clear=" << keys.size() << ", left=" << table_.size();
    return size;
  }

 //只应调用偏特化版本
 inline void GetKey(const int64_t* keys, int index, KeyType* result) {
   throw std::invalid_argument("GetKey for HashMap base should not be called");    
 }

 NonCocurrentHashTable* NewBlackList() {
   black_list_.reset(new NonCocurrentHashTable);
   return black_list_.get();
 }

 NonCocurrentHashTable* NewWhiteList() {
   white_list_.reset(new NonCocurrentHashTable);
   return white_list_.get();
 }

 NonCocurrentHashTable* GetBlackList() {
   return black_list_.get();
 }

 NonCocurrentHashTable* GetWhiteList() {
   return white_list_.get();
 }

 size_t FilterByBlackList() {
   size_t size = 0;
   for (auto key : *black_list_) {
     size += table_.unsafe_erase(key);
   }
   return size;
 }

 size_t FilterByWhiteList() {
   HashTable new_table;
   std::atomic<size_t> size(0);
   tbb::parallel_for_each(begin(table_), end(table_), [&](const std::pair<KeyType, size_t>& pr) {
     if (white_list_->find(pr.first) != white_list_->end()) {
       new_table.insert(pr);
     } else {
       size++;
     }
   });
   table_ = std::move(new_table);
   return size;
 }

 private:
  HashTable table_;
  std::unique_ptr<NonCocurrentHashTable> black_list_, white_list_;
  QRWLock lock_;
};

template<>
inline void HashMapImpl<int64_t>::GetKey(const int64_t* keys, int index, int64_t* result) {
  *result = keys[index];
}

template<>
inline void HashMapImpl<Hash128Key>::GetKey(const int64_t* keys, int index, Hash128Key* result) {
  result->hash1 = keys[2*index];
  result->hash2 = keys[2*index+1];
}

} //ps

#endif //PS_COMMON_HASHMAP_H
