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

#include "xdl/core/lib/thread_pool.h"

namespace xdl {

ThreadPool::ThreadPool(size_t threads)
  : thread_pool_(threads) {}

void ThreadPool::Schedule(const std::function<void()>& func) {
  thread_pool_.Schedule(func);
}

ThreadPool* ThreadPool::Global() {
  static ThreadPool tp(std::thread::hardware_concurrency());
  return &tp;
}

}  // namespace xdl

