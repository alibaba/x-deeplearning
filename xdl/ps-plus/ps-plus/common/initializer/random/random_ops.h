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

#ifndef PS_COMMON_INITIALIZER_RANDOM_RANDOM_OP_H
#define PS_COMMON_INITIALIZER_RANDOM_RANDOM_OP_H

#include "random_distributions.h"
#include "guarded_philox_random.h"

namespace ps {

template <class Distribution>
struct FillPhiloxRandom {
  typedef typename Distribution::ResultElementType T;
  void operator()(PhiloxRandom gen, 
                  T* data, 
                  int64 size, 
                  Distribution dist);
};

// A class to fill a specified range of random groups
template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomTask;

// Specialization for distribution that takes a fixed number of samples for
// each output.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  static void Run(PhiloxRandom gen, T* data, int64 size,
                  int64 start_group, int64 limit_group, Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    gen.Skip(start_group);
    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    for (int64 index = start_group; index < limit_group_full; ++index) {
      auto samples = dist(&gen);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    // If there are any remaining elements that need to be filled, process them
    if (limit_group_full < limit_group) {
      int64 remaining_size = size - limit_group_full * kGroupSize;
      auto samples = dist(&gen);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
  }
};

// Specialization for distribution that takes a variable number of samples for
// each output. This will be slower due to the generality.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  static const int64 kReservedSamplesPerOutput = 256;
  static void Run(PhiloxRandom base_gen, T* data, int64 size,
                  int64 start_group, int64 limit_group, Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;
    static const int kGeneratorSkipPerOutputGroup =
      kGroupSize * kReservedSamplesPerOutput /
      PhiloxRandom::kResultElementCount;

    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    int64 group_index;
    for (group_index = start_group; group_index < limit_group_full;
         ++group_index) {
      // Reset the generator to the beginning of the output group region
      // This is necessary if we want the results to be independent of order
      // of work
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      auto samples = dist(&single_samples);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    // If there are any remaining elements that need to be filled, process them
    if (limit_group_full < limit_group) {
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      int64 remaining_size = size - limit_group_full * kGroupSize;
      auto samples = dist(&single_samples);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
  }
};

// Partial specialization for CPU to fill the entire region with randoms
// It splits the work into several tasks and run them in parallel
template <typename Distribution>
void FillPhiloxRandom<Distribution>::operator()(PhiloxRandom gen,
                                                typename Distribution::ResultElementType* data, 
                                                int64 size,
                                                Distribution dist) {
  const int kGroupSize = Distribution::kResultElementCount;
  int64 limit_group = (size + kGroupSize - 1) / kGroupSize;
  FillPhiloxRandomTask<
    Distribution,
    Distribution::kVariableSamplesPerOutput>::Run(gen, data, size,
                                                  0, limit_group, dist);
};

class PhiloxRandomOp {
 public:
  explicit PhiloxRandomOp(uint64_t seed1, uint64_t seed2) {    
    generator_.Init(seed1, seed2);
  }

  template <typename Distribution>
  void Fill(size_t size, void *buf) {
    // add multi-thread support
    typedef typename Distribution::ResultElementType T;
    FillPhiloxRandom<Distribution>()(
        generator_.ReserveRandomOutputs(size, 256),
        reinterpret_cast<T*>(buf), size, 
        Distribution());
  }

 private:
  GuardedPhiloxRandom generator_;
};

}  // namespace ps

#endif // PS_COMMON_INITIALIZER_RANDOM_RANDOM_OP_H
