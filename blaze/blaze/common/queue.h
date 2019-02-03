/*
 * \file queue.h
 * \brief The message queue
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <queue>
#include <mutex>
#include <utility>

namespace blaze {

template <typename T>
class Queue {
 public:
  Queue() { exit_.store(false); }

  // Push an element into the queue, the function is based on move semantics.
  void Push(T item) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.push(item);
    empty_condition_.notify_one();
  }

  // Pop an element from the queue, if the queue is empty, thread call pop would
  // be blocked.
  bool Pop(T& result) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (buffer_.empty() && !exit_) {
      empty_condition_.wait(lock);
    }
    if (buffer_.empty()) return false;
    result = std::move(buffer_.front());
    buffer_.pop();
    return true;
  }

  // Thread will not be blocked, Return false if queue is empty
  bool TryPop(T& result) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (buffer_.empty()) {
      empty_condition_.wait_for(lock, std::chrono::milliseconds(1));
      if (buffer_.empty()) return false;
    }
    result = std::move(buffer_.front());
    buffer_.pop();
    return true;
  }

  // Get the front element from the queue, if the queue is empty, thread who
  // call front would be blocked.
  bool Front(T& result) {
    std::unique_lock<std::mutex> lock(mutex_);
    empty_condition_.wait(lock, [this]{ return !buffer_.empty() || exit_; });
    if (buffer_.empty()) return false;
    result = buffer_.front();
    return true;
  }

  // Get the number of elements in the queue
  int Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(buffer_.size());
  }

  // Whether queue is empty or not
  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.empty();
  }

  // Exit queue, awake all threads blocked by the queue
  void Exit() {
    std::lock_guard<std::mutex> lock(mutex_);
    exit_.store(true);
    empty_condition_.notify_all();
  }

  // Whether alive
  bool Alive() {
    std::lock_guard<std::mutex> lock(mutex_);
    return exit_ == false;
  }

 protected:
  std::queue<T> buffer_;
  mutable std::mutex mutex_;
  std::condition_variable empty_condition_;
  std::atomic_bool exit_;

  Queue(const Queue&);
  void operator=(const Queue&);
};

}  // namespace blaze
