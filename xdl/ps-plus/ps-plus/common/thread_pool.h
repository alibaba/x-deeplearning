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

#ifndef PS_PLUS_COMMON_THREAD_POOL_H
#define PS_PLUS_COMMON_THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <cstring>
#include "ps-plus/common/status.h"
#include "tbb/parallel_for.h"

namespace ps {

class ThreadPool {
public:
  ThreadPool(size_t threads);
  ~ThreadPool();
  void Schedule(const std::function<void()>& func);

  static ThreadPool* Global();
private:
  void Loop();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};

inline void QuickMemcpy(void* dest, const void* src, size_t count) {
  const size_t block_size = 1 << 30; // 1G
  if (count < block_size * 2) {
    memcpy(dest, src, count);
  } else {
    char* dest_ptr = (char*)dest;
    const char* src_ptr = (char*)src;
    std::promise<bool> ok;
    std::atomic<size_t> counter(count / block_size);
    while (count > 0) {
      size_t s = count < block_size * 2 ? count : block_size;
      ThreadPool::Global()->Schedule([=, &ok, &counter]{
        memcpy(dest_ptr, src_ptr, s);
        if (--counter == 0) {
          ok.set_value(true);
        }
      });
      dest_ptr += s;
      src_ptr += s;
      count -= s;
    }
    ok.get_future().wait();
  }
}

struct Range {
  size_t begin;
  size_t end;
};

inline Status MultiThreadDo(size_t size, const std::function<Status(const Range&)>& func, size_t block_size = 500) {
  static size_t thread_nums = std::thread::hardware_concurrency();
  if (size == 0) {
    return Status::Ok();
  }
  if (size < block_size) {
    Range range{.begin=0, .end=size};
    return func(range);
  } else {
    std::promise<bool> ok;
    if (size / block_size > thread_nums) {
      block_size = size / thread_nums;
    }
    size_t round = size / block_size;
    if (size % block_size != 0) {
      ++round;
    }
    std::atomic<size_t> counter(round);
    block_size = size / round;
    if (size % round != 0) {
      ++block_size;
    }
    Status st = Status::Ok();
    for (size_t i = 0; i < round; i++) {
      ThreadPool::Global()->Schedule([&, i]{
            Range range{.begin=i*block_size, .end=std::min(size, (i+1)*block_size)};
            Status ret = func(range);
            if (!ret.IsOk()) {
              st = ret;
            }
            if (--counter == 0) {
              ok.set_value(true);
            }});
    }
    ok.get_future().wait();
    return st;
  }
}

inline Status MultiThreadDoTBB(size_t size, const std::function<Status(const Range&)>& func, size_t block_size = 1) {
  static size_t thread_nums = std::thread::hardware_concurrency();
  if (size == 0) {
    return Status::Ok();
  }
  if (size < block_size) {
    Range range{.begin=0, .end=size};
    return func(range);
  } else {
    Status st = Status::Ok();
    parallel_for(tbb::blocked_range<size_t>(0, size), [&](tbb::blocked_range<size_t>& r) {
          Range range{.begin=r.begin(), .end=r.end()};
          Status ret = func(range);
          if (!ret.IsOk()) {
            st = ret;
          }
        });
    return st;
  }
}

}

#endif

