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

#include "bloom_filter.h"

#include <cmath>
#include <set>
#include <random>
#include <limits>

#include "murmurhash.h"
#include "ps-plus/common/logging.h"

namespace ps {

template <typename CType>
CountingBloomFilter<CType>::CountingBloomFilter(double fpp,
                                                uint64_t element_size)
  : false_positive_probability_(fpp),
    element_size_(element_size) {
  bucket_size_ = static_cast<uint64_t>(std::ceil(
          std::abs(std::log(fpp) / std::pow(std::log(2.0), 2)) * element_size));
  hash_function_number_ = static_cast<uint32_t>(std::ceil(
          std::abs(std::log2(fpp))));
  std::set<uint32_t> seeds;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dist(0, std::numeric_limits<uint32_t>::max());
  while (seeds.size() != hash_function_number_) {
    seeds.insert(dist(gen));
  }
  for (const auto& seed : seeds) {
    hash_functions_.push_back(MurmurHash(seed));
  }
  buf_.resize(bucket_size_, CType{});
  LOG(INFO) << "optimal bloom filter parameters: false_positive_probability=" << false_positive_probability_ \
            << " element_size=" << element_size_ << ", bucket_size=" << bucket_size_ << " hash_function_number=" << hash_function_number_;
}

template <typename CType>
void CountingBloomFilter<CType>::Insert(const void* key, int len) {
  if (key == nullptr) return;
  uint64_t mur_res[2];
  for (auto&& fn : hash_functions_) {
    fn(key, len, mur_res);
    uint64_t index = mur_res[1] % bucket_size_;
    if (buf_[index] != std::numeric_limits<CType>::max()) {
      buf_[index] += 1;
    }
  }
}

template <typename CType>
bool CountingBloomFilter<CType>::Exists(const void* key, int len,
                                        uint32_t max_count) const {
  if (max_count == 0) return true;
  if (key == nullptr) return false;
  uint64_t mur_res[2];
  for (auto&& fn : hash_functions_) {
    fn(key, len, mur_res);
    uint64_t index = mur_res[1] % bucket_size_;
    if (buf_[index] < max_count) return false;
  }
  return true;
}

template <typename CType>
bool CountingBloomFilter<CType>::InsertedLookup(const void* key, int len,
                                                uint32_t max_count) {
  Insert(key, len);
  return Exists(key, len, max_count);
}

BloomFilterBase* GlobalBloomFilter::Instance() {
  if (filter == nullptr) {
    lock.lock();
    if (filter == nullptr) {
      if (max_threthold < 240) {
        filter = new CountingBloomFilter<uint8_t>();
      } else {
        filter = new CountingBloomFilter<uint16_t>();
      }
    }
    lock.unlock();
  }
  return filter;
}

void GlobalBloomFilter::SetThrethold(int32_t threthold) {
  if (threthold > max_threthold) {
    lock.lock();
    if (threthold > max_threthold) {
      max_threthold = threthold;
    }
    lock.unlock();
  }
}

BloomFilterBase* GlobalBloomFilter::filter = nullptr;
int32_t GlobalBloomFilter::max_threthold = 0;
std::mutex GlobalBloomFilter::lock;

}  // namespace ps
