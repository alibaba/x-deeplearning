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

#ifndef PS_COMMON_INITIALIZER_RANDOM_PHILOX_RANDOM_H
#define PS_COMMON_INITIALIZER_RANDOM_PHILOX_RANDOM_H

#include <stdlib.h>
#include <math.h>
#include <cstdint>

namespace ps {

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

static const uint8_t kuint8max = ((uint8_t)0xFF);
static const uint16_t kuint16max = ((uint16_t)0xFFFF);
static const uint32_t kuint32max = ((uint32_t)0xFFFFFFFF);
static const uint64_t kuint64max = ((uint64_t)0xFFFFFFFFFFFFFFFFull);
static const int8_t kint8min = ((int8_t)~0x7F);
static const int8_t kint8max = ((int8_t)0x7F);
static const int16_t kint16min = ((int16_t)~0x7FFF);
static const int16_t kint16max = ((int16_t)0x7FFF);
static const int32_t kint32min = ((int32_t)~0x7FFFFFFF);
static const int32_t kint32max = ((int32_t)0x7FFFFFFF);
static const int64_t kint64min = ((int64_t)~0x7FFFFFFFFFFFFFFFll);
static const int64_t kint64max = ((int64_t)0x7FFFFFFFFFFFFFFFll);

template <typename T, int ElementCount>
class Array {
 public:
  Array() {
    for (int i = 0; i < ElementCount; ++i) {
      data_[i] = T(0);
    }
  }

  const T& operator[](int index) const {
    return data_[index];
  }

  T& operator[](int index) { return data_[index]; }

  size_t size() const { return ElementCount; }

 private:
  T data_[ElementCount];
};

// A class that encapsulates all the states for a random number generator using
// the philox_4x32_10 algorithm. Each invocation returns a 128-bit random bits
// in the form of four uint32_t.
// There are multiple variants of this algorithm, we picked the 4x32_10 version
// that is most suited for our applications.
// Since this class is meant to be copied between CPU to GPU, it maintains a
// value semantics.
//
// For example: To use this class and populate an array of 1024 randoms on CPU
// with two threads,
//
//  void Fill(PhiloxRandom rnd, uint32_t* output, int start, int limit) {
//    assert(start % 4 == 0);
//    assert(limit % 4 == 0);
//    rnd.Skip(start / 4);
//    for (int i = start; i < limit; i += 4) {
//      auto sample = rnd();
//      ... copy sample[0..3] to output[i..i+3]
//    }
//  }
//
//  PhiloxRandom rng(seed);
//  PhiloxRandom rng_copy = rng;
//  rng.Skip(1000/4);
//
//  ... schedule Fill(rng_copy, output, 0, 512) in thread 1;
//  ... schedule Fill(rng_copy, output, 512, 1024) in thread 2;
//  ... wait for thread 1 & 2 to finish executing Fill().
//

class PhiloxRandom {
 public:
  typedef Array<uint32_t, 4> ResultType;
  typedef uint32_t ResultElementType;
  // The number of elements that will be returned.
  static const int kResultElementCount = 4;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 10;

  PhiloxRandom() {}

  explicit PhiloxRandom(uint64 seed) {
    key_[0] = static_cast<uint32_t>(seed);
    key_[1] = static_cast<uint32_t>(seed >> 32);
  }

  explicit PhiloxRandom(uint64 seed_lo, uint64 seed_hi) {
    key_[0] = static_cast<uint32_t>(seed_lo);
    key_[1] = static_cast<uint32_t>(seed_lo >> 32);
    counter_[2] = static_cast<uint32_t>(seed_hi);
    counter_[3] = static_cast<uint32_t>(seed_hi >> 32);
  }

  // Skip the specified number of samples of 128-bits in the current stream.
  inline void Skip(uint64 count) {
    const uint32_t count_lo = static_cast<uint32_t>(count);
    uint32_t count_hi = static_cast<uint32_t>(count >> 32);

    counter_[0] += count_lo;
    if (counter_[0] < count_lo) {
      ++count_hi;
    }

    counter_[1] += count_hi;
    if (counter_[1] < count_hi) {
      if (++counter_[2] == 0) {
        ++counter_[3];
      }
    }
  }

  // Returns a group of four random numbers using the underlying Philox
  // algorithm.
  inline ResultType operator()() {
    ResultType counter = counter_;
    Key key = key_;

    // Run the single rounds for ten times. Manually unrolling the loop
    // for better performance.
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);

    SkipOne();

    return counter;
  }

 private:
  // The type for the 64-bit key stored in the form of two 32-bit uint
  // that are used in the diffusion process.
  typedef Array<uint32_t, 2> Key;

  // We use the same constants as recommended by the original paper.
  static const uint32_t kPhiloxW32A = 0x9E3779B9;
  static const uint32_t kPhiloxW32B = 0xBB67AE85;
  static const uint32_t kPhiloxM4x32A = 0xD2511F53;
  static const uint32_t kPhiloxM4x32B = 0xCD9E8D57;

  // Helper function to skip the next sample of 128-bits in the current stream.
  inline void SkipOne() {
    if (++counter_[0] == 0) {
      if (++counter_[1] == 0) {
        if (++counter_[2] == 0) {
          ++counter_[3];
        }
      }
    }
  }

  // Helper function to return the lower and higher 32-bits from two 32-bit
  // integer multiplications.
  inline static void MultiplyHighLow(uint32_t a, uint32_t b, uint32_t* result_low,
                                     uint32_t* result_high) {
    const uint64 product = static_cast<uint64>(a) * b;
    *result_low = static_cast<uint32>(product);
    *result_high = static_cast<uint32>(product >> 32);      
  }

  // Helper function for a single round of the underlying Philox algorithm.
  inline static ResultType ComputeSingleRound(
      const ResultType& counter, const Key& key) {
    uint32_t lo0;
    uint32_t hi0;
    MultiplyHighLow(kPhiloxM4x32A, counter[0], &lo0, &hi0);

    uint32_t lo1;
    uint32_t hi1;
    MultiplyHighLow(kPhiloxM4x32B, counter[2], &lo1, &hi1);

    ResultType result;
    result[0] = hi1 ^ counter[1] ^ key[0];
    result[1] = lo1;
    result[2] = hi0 ^ counter[3] ^ key[1];
    result[3] = lo0;
    return result;
  }

  inline void RaiseKey(Key* key) {
    (*key)[0] += kPhiloxW32A;
    (*key)[1] += kPhiloxW32B;
  }

 private:
  ResultType counter_;
  Key key_;
};

}  // namespace ps

#endif  // PS_COMMON_INITIALIZER_RANDOM_PHILOX_RANDOM_H
