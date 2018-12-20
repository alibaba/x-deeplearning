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

#include "guarded_philox_random.h"

#include "random.h"

using namespace std;

namespace ps {

void GuardedPhiloxRandom::Init(int64 seed, int64 seed2) {
  {
    lock_guard<mutex> lock(mu_);
    if (initialized_) {
      return;
    }
  }

  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, use completely random seeds.
    seed = Random::New64();
    seed2 = Random::New64();
  }

  lock_guard<mutex> lock(mu_);
  generator_ = PhiloxRandom(seed, seed2);
  initialized_ = true;
}

PhiloxRandom GuardedPhiloxRandom::ReserveSamples128(int64 samples) {
  lock_guard<mutex> lock(mu_);
  auto local = generator_;
  generator_.Skip(samples);
  return local;
}

}  // ps
