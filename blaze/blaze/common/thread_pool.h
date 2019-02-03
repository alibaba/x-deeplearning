/*
 * \file thread_pool.h
 * \desc The thread pool and thread executor
 */
#pragma once

#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <queue>
#include <thread>
#include <vector>

#include "blaze/common/common_defines.h"
#include "blaze/common/queue.h"

namespace blaze {

/*!
 * \brief Thread pool.
 */
class ThreadPool {
 public:
  /*!
   * \brief Constructor takes function to run.
   * \param size size of the thread pool.
   * \param func the function to run on the thread pool.
   */
  explicit ThreadPool(size_t size, std::function<void(int)> func)
      : worker_threads_(size) {
    int idx = 0;
    for (auto& i : worker_threads_) {
      i = std::thread(func, idx++);
    }
  }
  ~ThreadPool() noexcept(false) {
    for (auto&& i : worker_threads_) { i.join(); }
  }
  
  size_t Size() const {
    return worker_threads_.size();
  }

 private:
  /*!
   * \brief Worker threads.
   */
  std::vector<std::thread> worker_threads_;
  /*!
   * \brief Disallow default construction.
   */
  ThreadPool() = delete;
  /*!
   * \brief Disallow copy construction and assignment.
   */
  DISABLE_COPY_AND_ASSIGN(ThreadPool);
};

class ThreadExecutor {
 public:
  using Task = std::function<void()>;
  
  ThreadExecutor(size_t size = 10) {
    size = size < 1 ? 1 : size;
    for (size_t i = 0; i< size; ++i) {
      pool_.emplace_back(&ThreadExecutor::schedule, this);
    }
  }

  ~ThreadExecutor() {
    for(std::thread& thread : pool_){
      thread.join();
    }
  }

  void shutdown() {
    while (!queue_.Empty()) usleep(10);
    queue_.Exit();
  }

  template<class F, class... Args>
  auto commit(F&& f, Args&&... args) ->std::future<decltype(f(args...))> {
    using ResType =  decltype(f(args...));
    auto task = std::make_shared<std::packaged_task<ResType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
    {
      queue_.Push([task]() { (*task)(); });
    }
    std::future<ResType> future = task->get_future();
    return future;
  }

 private:
  void schedule() {
    while (true) {
      Task task;
      if (!queue_.Pop(task)) break;
      task();
    }
  }

 private:
  std::vector<std::thread> pool_;
  Queue<Task> queue_;
};

}  // namespace blaze 
