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

#ifndef XDL_CORE_LIB_THREAD_POOL_H_
#define XDL_CORE_LIB_THREAD_POOL_H_

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "unsupported/Eigen/CXX11/ThreadPool"

namespace Eigen {
template <typename Env> using ThreadPoolTempl = NonBlockingThreadPoolTempl<Env>;
typedef NonBlockingThreadPool ThreadPool;
}

namespace xdl {

class ThreadPool {
 public:
  explicit ThreadPool(size_t threads);
  void Schedule(const std::function<void()>& func);
  int CurrentThreadId() const {
    return thread_pool_.CurrentThreadId();
  }

  static ThreadPool* Global();
 private:
  Eigen::ThreadPool thread_pool_;
};

}  // namespace xdl

#endif  // XDL_CORE_LIB_THREAD_POOL_H_

