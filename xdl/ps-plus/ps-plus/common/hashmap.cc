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

#include <iostream>
#include <algorithm>
#include <unordered_set>
#include "hashmap.h"

namespace ps {

HashMap::HashMap() : offset_(0), urd(0.0, 1.0), max_count_(0) {
}

HashMap::~HashMap() {
}

bool HashMap::FloatEqual(float v1, float v2) {
  return (v1 <= (v2 + FLOAT_EPSILON)) && (v1 >= (v2 - FLOAT_EPSILON));
}
  
void HashMap::SetBloomFilterThrethold(int32_t max_count) {
  max_count_ = max_count;
}

std::ostream& operator<<(std::ostream& os, const Hash128Key& key) {
  os << key.hash1 << "," << key.hash2;
  return os;
}

const float HashMap::FLOAT_EPSILON = 1.192092896e-07f;
const size_t HashMap::NOT_ADD_ID = -2;
} //ps
