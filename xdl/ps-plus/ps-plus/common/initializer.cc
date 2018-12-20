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
    char* ptr = (char*)data;
    std::promise<bool> ok;
    std::atomic<size_t> counter(size / block_size);
    while (size > 0) {
      size_t s = size < block_size * 2 ? size : block_size;
      ThreadPool::Global()->Schedule([=, &ok, &counter]{
        Init(ptr, type, s);
        if (--counter == 0) {
          ok.set_value(true);
        }
      });
      ptr += SizeOfType(type) * s;
      size -= s;
    }
    ok.get_future().wait();
  }
}

}


