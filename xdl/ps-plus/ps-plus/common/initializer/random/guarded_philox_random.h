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

#ifndef PS_COMMON_INITIALIZER_RANDOM_GUARDED_PHILOX_RANDOM_H
#define PS_COMMON_INITIALIZER_RANDOM_GUARDED_PHILOX_RANDOM_H

#include "philox_random.h"

#include <random>
#include <mutex>

namespace ps {

// A thread safe wrapper around a Philox generator.  Example usage:
//
//   GuardedRandomPhilox generator;
//   generator.Init(context);
//
//   // In thread safe code
//   const int samples = ...;
//   auto local_generator = generator.ReserveSamples128(samples);
//   for (int i = 0; i < samples; i++)
//     Array<uint32, 4> sample = local_generator();
//     // Use sample
//   }
//

class GuardedPhiloxRandom {
 public:
  // Must call Init to finish initialization
  GuardedPhiloxRandom() : initialized_(false) {}

  // Initialize with given seeds.
  void Init(int64 seed, int64 seed2);

  // Reserve a certain number of 128-bit samples.
  // This function is thread safe.  The returned generator is valid for the
  // given number of samples, and can be used without a lock.
  PhiloxRandom ReserveSamples128(int64 samples);

  // Reserve a certain number of 32-bit samples.
  PhiloxRandom ReserveSamples32(int64 samples) {
    return ReserveSamples128((samples + 3) / 4);
  }

  // Reserve enough random samples in the generator for the given output count.
  PhiloxRandom ReserveRandomOutputs(int64 output_count,
                                    int multiplier) {
    int64 conservative_sample_count = output_count * multiplier;
    return ReserveSamples128(conservative_sample_count);
  }

 private:
  std::mutex mu_;
  PhiloxRandom generator_;
  bool initialized_;
};

}  // ps

#endif  // PS_COMMON_INITIALIZER_RANDOM_GUARDED_PHILOX_RANDOM_H
