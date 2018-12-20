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

#ifndef PS_COMMON_INITIALIZER_RANDOM_RANDOM_H
#define PS_COMMON_INITIALIZER_RANDOM_RANDOM_H

#include <random>

namespace ps {

class Random {
 public:
  static uint64_t New64();
  static std::mt19937_64* InitRng();
  static void GetSeed(int* seed,
                      int* seed1, 
                      int* seed2);
 private:
  struct ThreadRandom {
    ThreadRandom() : engine(std::random_device()()) {}
    std::mt19937 engine;
  };
  static thread_local ThreadRandom tr_;
  static const int DEFAULT_SEED1 = 87654321;
};

}  // namespace ps

#endif // PS_COMMON_INITIALIZER_RANDOM_RANDOM_H
