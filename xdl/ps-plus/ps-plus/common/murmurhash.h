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

#ifndef PS_PLUS_COMMON_MURMURHASH_H_
#define PS_PLUS_COMMON_MURMURHASH_H_

#include <cstdint>

namespace ps {

class MurmurHash {
 public:
  MurmurHash(uint32_t seed) : seed_(seed) {}
  void operator()(const void* key, int len, void* out);
 private:
  uint32_t seed_;
};

}  // namespace ps

#endif  // PS_PLUS_COMMON_MURMURHASH_H_
