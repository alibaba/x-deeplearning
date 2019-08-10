/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef XDL_CORE_LIB_CONCURRENT_QUEUE_H_
#define XDL_CORE_LIB_CONCURRENT_QUEUE_H_

#include <queue>
#include <mutex>
#include <condition_variable>

namespace xdl {

template <typename T>
class ConcurrentQueue {
 public:
  using Callback = std::function<void(T)>;
  using Waiter = std::function<void()>;
  ConcurrentQueue() : waiter_size_(0) {}
  void Pop(Callback cb) {
    T val;
    bool run = false;
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (!queue_.empty()) {
        val = queue_.front();
        queue_.pop_front();
        run = true;
      } else {
        cbs_.push_back(cb);
      }
    }
    if (run) {
      cb(val);
    }
  }
  void Push(T val) {
    Callback cb;
    bool run = false;
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (!cbs_.empty()) {
        cb = cbs_.front();
        cbs_.pop_front();
        run = true;
      } else {
        queue_.push_back(val);
      }
      if (waiters_.empty()) {
        waiter_size_++;
      } else {
        waiters_.front()();
        waiters_.pop_front();
      }
    }
    if (run) {
      cb(val);
    }
  }
  void Wait(Waiter waiter) {
    std::unique_lock<std::mutex> lock(mu_);
    if (waiter_size_ > 0) {
      waiter_size_--;
      waiter();
    } else {
      waiters_.push_back(waiter);
    }
  }
 private:
  std::mutex mu_;
  std::deque<T> queue_;
  std::deque<Callback> cbs_;
  std::deque<Waiter> waiters_;
  int waiter_size_;
};

}  // namespace xdl

#endif  // XDL_CORE_LIB_CONCURRENT_QUEUE_H_
