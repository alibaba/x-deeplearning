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

#include "ps-plus/common/initializer.h"
#include "ps-plus/common/thread_pool.h"
#include "tbb/parallel_for.h"
#include <future>

namespace ps {

bool Initializer::Accept(DataType type) {
  return true;
}

void Initializer::MultiThreadInit(void* data, DataType type, size_t size) {
  static const size_t block_size = 1 << 15; // 32K
  if (size < block_size * 2) {
    Init(data, type, size);
  } else {
    parallel_for(tbb::blocked_range<size_t>(0, size), [&](tbb::blocked_range<size_t>& r) {
          size_t start = r.begin();
          size_t end = r.end();
          Init(data + (start * SizeOfType(type)), type, end-start);
        });
  }
}

}


