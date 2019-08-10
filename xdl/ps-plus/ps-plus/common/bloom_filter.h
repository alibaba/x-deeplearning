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

#ifndef PS_PLUS_COMMON_BLOOM_FILTER_H_
#define PS_PLUS_COMMON_BLOOM_FILTER_H_

#include <mutex>
#include <vector>
#include <memory>
#include <functional>

namespace ps {

class BloomFilterBase {
 public: 
  virtual void Insert(const void* key, int len) = 0;
  virtual bool Exists(const void* key, int len, uint32_t max_count=1u) const = 0;
  virtual bool InsertedLookup(const void* key, int len, uint32_t max_count=1u) = 0;
};

template <typename CType>
class CountingBloomFilter : public BloomFilterBase {
 public:
  using Hasher = std::function<void(const void*, int, void*)>;
  CountingBloomFilter(double fpp=0.01, uint64_t element_size=1000000000);
  ~CountingBloomFilter() {}
  virtual void Insert(const void* key, int len);
  virtual bool Exists(const void* key, int len, uint32_t max_count=1u) const;
  virtual bool InsertedLookup(const void* key, int len, uint32_t max_count=1u);
  double false_positive_probability() const {
    return false_positive_probability_;
  }
  uint64_t element_size() const {
    return element_size_;
  }
  uint64_t bucket_size() const {
    return bucket_size_;
  }
  uint32_t hash_function_number() const {
    return hash_function_number_;
  }
 private:
  CountingBloomFilter(const CountingBloomFilter&) = delete;
  CountingBloomFilter& operator=(const CountingBloomFilter&) = delete;

  double false_positive_probability_;
  uint64_t element_size_;
  uint64_t bucket_size_;
  uint32_t hash_function_number_;
  std::vector<Hasher> hash_functions_;

  std::vector<CType> buf_;
};

class GlobalBloomFilter {
 public: 
  static void SetThrethold(int32_t threthold);
  static BloomFilterBase* Instance();
 private:
  GlobalBloomFilter(const GlobalBloomFilter&) = delete;
  GlobalBloomFilter& operator=(const GlobalBloomFilter&) = delete;
  static int32_t max_threthold;
  static BloomFilterBase* filter;
  static std::mutex lock;
};

template class CountingBloomFilter<uint8_t>;
template class CountingBloomFilter<uint16_t>;

}  // namespace ps

#endif  // PS_PLUS_COMMON_BLOOM_FILTER_H_
