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

#include "random.h"

#include <mutex>
#include <limits>

using namespace std;

namespace ps {

std::mt19937_64* Random::InitRng() {
  std::random_device device("/dev/urandom");
  return new std::mt19937_64(device());
}

uint64_t Random::New64() {
  static std::mt19937_64* rng = InitRng();
  static std::mutex mu;
  lock_guard<mutex> l(mu);
  return (*rng)();
}

void Random::GetSeed(int* seed,
                     int* seed1,
                     int* seed2) {
  /*
  if (seed != nullptr) {
    *seed1 = DEFAULT_SEED1;
    *seed2 = *seed;
  } else {
    *seed1 = 0;
    *seed2 = numeric_limits<int>::max();
  }
  */
  *seed1 = tr_.engine();
  *seed2 = tr_.engine();
}

thread_local Random::ThreadRandom Random::tr_;

} // namespace ps
