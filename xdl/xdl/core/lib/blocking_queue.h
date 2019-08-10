/*
 * Copyright 1999-2018 Alibaba Group.
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
#ifndef XDL_CORE_LIB_BLOCKING_QUEUE_H_
#define XDL_CORE_LIB_BLOCKING_QUEUE_H_

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <deque>
#include <chrono>
#include <unordered_map>
#include <set>
#include <string>
#include <type_traits>

#include "xdl/core/utils/logging.h"

namespace xdl {

template <typename Task>
class BlockingQueue {
 public:
  using Compare = std::function<bool(const Task&, const Task&)>;

  BlockingQueue(size_t capacity = 10, float ratio = 1.0, Compare comp = nullptr) :
      capacity_(capacity), ratio_(ratio), comp_(comp) { }

  /*!\brief destruction of blocking queue */
  virtual ~BlockingQueue() { }

  /*!\brief Enqueue the queue */
  void Enqueue(const Task& task, bool force = false) {
    std::unique_lock<std::mutex> lck(mutex_);
    size_t max_cap = force ? capacity_ : capacity_ * ratio_;
    while (tasks_.size() >= max_cap) {
      cv_full_.wait(lck);
    }
    tasks_.push_back(task);
    if (comp_ != nullptr) {
      std::push_heap(tasks_.begin(), tasks_.end(), comp_);
    }
    cv_empty_.notify_all();
  }
  void ForceEnqueue(const Task& task) {
    std::unique_lock<std::mutex> lck(mutex_);
    tasks_.push_back(task);
    if (comp_ != nullptr) {
      std::push_heap(tasks_.begin(), tasks_.end(), comp_);
    }
    cv_empty_.notify_all();
  }

  /*!\brief Enqueue with elapse timwait */
  bool TryEnqueue(const Task& task, uint32_t timewait) {
    std::unique_lock<std::mutex> lck(mutex_);
    size_t max_cap = capacity_ * ratio_;
    if (tasks_.size() >= max_cap) cv_full_.wait_for(lck, std::chrono::milliseconds(timewait));
    if (tasks_.size() >= max_cap) return false;
    tasks_.push_back(task);
    if (comp_ != nullptr) {
      std::push_heap(tasks_.begin(), tasks_.end(), comp_);
    }
    cv_empty_.notify_all();
    return true;
  }
  /*!\brief Dequeue the queue */
  Task Dequeue() {
    std::unique_lock<std::mutex> lck(mutex_);
    while (tasks_.empty()) {
      cv_empty_.wait(lck);
    }
    Task task = tasks_.front();
    if (comp_ != nullptr) {
      std::pop_heap(tasks_.begin(), tasks_.end(), comp_);
      tasks_.pop_back();
    } else {
      tasks_.pop_front();
    }
    cv_full_.notify_all();
    return task;
  }
  /*!\brief Try Dequeue */
  bool TryDequeue(Task* task, uint32_t timewait) {
    std::unique_lock<std::mutex> lck(mutex_);
    if (tasks_.empty()) cv_empty_.wait_for(lck, std::chrono::milliseconds(timewait));
    if (tasks_.empty()) return false;
    *task = tasks_.front();
    if (comp_ != nullptr) {
      std::pop_heap(tasks_.begin(), tasks_.end(), comp_);
      tasks_.pop_back();
    } else {
      tasks_.pop_front();
    }
    cv_full_.notify_all();
    return true;
  }

  /*!\brief Eequeue front */
  void EnqueueFront(const Task& task) {
    XDL_CHECK(comp_ == nullptr) << " not for priority queue";
    std::unique_lock<std::mutex> lck(mutex_);
    size_t max_cap = capacity_;
    while (tasks_.size() >= max_cap) {
      cv_full_.wait(lck);
    }
    tasks_.push_front(task);
    cv_empty_.notify_all();
  }

  /*!\brief Return size */
  size_t Size() {
    std::unique_lock<std::mutex> lck(mutex_);
    return tasks_.size();
  }

  /*!\brief whether is full */
  bool Full(uint32_t timewait = 1) {
    std::unique_lock<std::mutex> lck(mutex_);
    if (tasks_.size() < capacity_) return false;
    cv_full_.wait_for(lck, std::chrono::milliseconds(timewait));
    return tasks_.size() >= capacity_;
  }
  
  /*!\brief whether is empty */
  bool Empty() {
    std::unique_lock<std::mutex> lck(mutex_);
    return tasks_.size() == 0;
  }

  /*!\brief clear queue */
  void Clear() {
    std::unique_lock<std::mutex> lck(mutex_);
    tasks_.clear();
  }
  /*!\brief clear and release */
  void ClearAndDelete(void (*deleter)(Task)) {
    std::unique_lock<std::mutex> lck(mutex_);
    while (!tasks_.empty()) {
      Task task = tasks_.front();
      deleter(task);
      tasks_.pop_front();
    }
  }

  void Travel(std::function<void(const Task &task, size_t i)> on_task) {
    std::unique_lock<std::mutex> lck(mutex_);
    size_t c = 0;
    for (auto it = tasks_.cbegin(); it != tasks_.cend(); ++it, ++c) {
      on_task(*it, c);
    }
  }

 protected:
  /// The capacity
  size_t capacity_;
  /// ratio of enqueue on capacity
  float ratio_ = 1.0;
  /// the tasks
  std::deque<Task> tasks_;
  /// the mutexs
  std::mutex mutex_;
  /// cv empty
  std::condition_variable cv_empty_;
  /// cv full
  std::condition_variable cv_full_;
  Compare comp_ = nullptr;
};


template <typename Task>
class MultiBlockingQueue {
 public:
  MultiBlockingQueue(size_t capacity = 10, float ratio = 0.5) :
      capacity_(capacity), ratio_(ratio) { }
  
  virtual ~MultiBlockingQueue() {
    std::unique_lock<std::mutex> lck(mutex_);
    for (auto& iter : kvs_) delete iter.second;
  }

  /*!\brief enqueue on name */
  void Enqueue(const std::string& name, const Task& task, bool force = false) {
    BlockingQueue<Task>* queue = QueueAt(name);
    queue->Enqueue(task, force);
  }
  /*!\brief force enqueue on name */
  void ForceEnqueue(const std::string& name, const Task& task) {
    BlockingQueue<Task>* queue = QueueAt(name);
    queue->ForceEnqueue(task);
  }

  /*!\brief try enqueue on name with elapse timewait */
  bool TryEnqueue(const std::string& name, const Task& task, uint32_t timewait) {
    BlockingQueue<Task>* queue = QueueAt(name);
    return queue->TryEnqueue(task, timewait);
  }
  /*!\brief Enqueue for each
   */
  void EnqueueForEach(const Task& task) {
    std::unique_lock<std::mutex> lck(mutex_);
    for (auto& iter : kvs_) {
      iter.second->Enqueue(task, true);
    }
  }

  /*!\brief dequeue on name
   * coupled with enqueue, A queue is implemented.
   */
  Task Dequeue(const std::string& name) {
    BlockingQueue<Task>* queue = QueueAt(name);
    return queue->Dequeue();
  }
  /*! \brief Try dequeue
   */
  bool TryDequeue(const std::string& name, Task* task, uint32_t timewait) {
    BlockingQueue<Task>* queue = QueueAt(name);
    return queue->TryDequeue(task, timewait);
  }

  /*!\brief enueue at front
   * couple with enqueue, A stack is implemented.
   */
  void EnqueueFront(const std::string& name, const Task& task) {
    BlockingQueue<Task>* queue = QueueAt(name);
    queue->EnqueueFront(task);
  }

  /*!\brief return size of name */
  size_t Size(const std::string& name) {
    BlockingQueue<Task>* queue = QueueAt(name);
    return queue->Size();
  }

  /*!\brief return queue num */
  size_t QueueNum() {
    std::unique_lock<std::mutex> lck(mutex_);
    return kvs_.size();
  }

  /*!\brief whether is fill */
  bool Full(const std::string& name, uint32_t timewait = 1) {
    BlockingQueue<Task>* queue = QueueAt(name);
    return queue->Full(timewait);
  }

  /*!\brief whether is empty */
  bool Empty(const std::string& name) {
    BlockingQueue<Task>* queue = QueueAt(name);
    return queue->Empty();
  }

  /*!\brief clear queue */
  void Clear(const std::string& name) {
    BlockingQueue<Task>* queue = QueueAt(name);
    queue->Clear();
  }
  /*!\brief clear and release queue */
  void ClearAndDelete(const std::string& name, void (*deleter)(Task)) {
    BlockingQueue<Task>* queue = QueueAt(name);
    queue->ClearAndDelete(deleter);
  }

  /*!\brief return names */
  std::set<std::string> Names() {
    std::unique_lock<std::mutex> lck(mutex_);
    std::set<std::string> names;
    for (auto& iter : kvs_) names.insert(iter.first);
    return names;
  }

 protected:
  /*!\brief queue on name */
  BlockingQueue<Task>* QueueAt(const std::string& name) {
    std::unique_lock<std::mutex> lck(mutex_);
    const auto& iter = kvs_.find(name);
    BlockingQueue<Task>* ret = nullptr;
    if (iter == kvs_.end()) {
      ret = new BlockingQueue<Task>(capacity_, ratio_);
      kvs_[name] = ret;
    } else {
      ret = iter->second;
    }
    return ret;
  }

  std::unordered_map<std::string, BlockingQueue<Task>*> kvs_;
  size_t capacity_;
  float ratio_;
  std::mutex mutex_;
};

}  // namespace xdl

#endif // XDL_CORE_BASE_BLOCKING_QUEUE_H_
