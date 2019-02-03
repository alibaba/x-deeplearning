/*
 * \file simple_scheduler.h 
 * \brief The simple scheduler is used for scheduling tasks in simple way
 */
#ifndef BLAZE_SCHEDULER_SIMPLE_SCHEDULER_H_
#define BLAZE_SCHEDULER_SIMPLE_SCHEDULER_H_

#include <memory>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "blaze/scheduler/scheduler.h"
#include "blaze/common/log.h"
#include "blaze/common/semaphore.h"

namespace blaze {

template <typename TaskType> 
class SimpleScheduler : public Scheduler {
 public:
  using ProcessFunc = std::function<void(std::unique_ptr<TaskType>)>;

  virtual ~SimpleScheduler();

  struct Options {
    // The name to use for the pool of threads
    std::string thread_pool_name = {"threads"}; 
    // The number of threads used to process tasks
    int num_threads = 10;
    // The queue capacity for pending tasks
    int queue_capacity = 100;
  };

  static bool Create(const Options& options,
      std::shared_ptr<SimpleScheduler<TaskType>>* scheduler);   

  void Schedule(std::unique_ptr<TaskType> task,
      std::shared_ptr<ProcessFunc> process_func);

  size_t size() const {
    return queue_.size();
  }

 private:
  struct QueueNode {
    QueueNode(std::unique_ptr<TaskType> t, std::shared_ptr<ProcessFunc> p)
      : task(std::move(t)), process_func(std::move(p)) {}

    std::unique_ptr<TaskType> task;
    std::shared_ptr<ProcessFunc> process_func; 
  };

  explicit SimpleScheduler(const Options& options);
  
  void ProcessBody();

  std::queue<QueueNode> queue_;
  size_t queue_capacity_; 
  std::mutex queue_mutex_;
  Semaphore empty_;
  Semaphore full_; 
  std::vector<std::unique_ptr<std::thread>> thread_pool_;
  volatile bool stop_running_;
};

template <typename TaskType>
bool SimpleScheduler<TaskType>::Create(const Options& options,
      std::shared_ptr<SimpleScheduler<TaskType>>* scheduler) {
  if (nullptr == scheduler) {
    LOG_ERROR("Input parameter scheduler is nullptr");
    return false;
  }  
  (*scheduler).reset(new SimpleScheduler(options));
  return true;  
}

template <typename TaskType>
SimpleScheduler<TaskType>::SimpleScheduler(const Options& options)
    : stop_running_(false) {
  // init queue and thread pool by input options 
  queue_capacity_ = options.queue_capacity;
  for (int i = 0; i < options.num_threads; ++i) {
    thread_pool_.emplace_back(new std::thread(
          [this] {this->ProcessBody();}));
  }
  empty_.Init(queue_capacity_);
}

template <typename TaskType>
SimpleScheduler<TaskType>::~SimpleScheduler() {
  stop_running_ = true;
  for (int i = 0; i < thread_pool_.size(); ++i) {
    // Mock full_.notify() to make ProcessBody wakeup and move forward
    full_.notify();
  }
  for (int i = 0; i < thread_pool_.size(); ++i) {
    thread_pool_[i]->join();
  }
  thread_pool_.clear();
}

template <typename TaskType>
void SimpleScheduler<TaskType>::Schedule(std::unique_ptr<TaskType> task,
    std::shared_ptr<ProcessFunc> process_func) {
  // first wait for empty space
  empty_.wait();
  // protect the update of queue_ 
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_.emplace(std::move(task), process_func);
    // release the lock
  }
  full_.notify();  
}

template <typename TaskType>
void SimpleScheduler<TaskType>::ProcessBody() {
  // in each thread, invoke ProcessFunc to process TaskType 
  while (!stop_running_) {
    full_.wait();
    if (stop_running_) {
      return;
    }
    // protect the update of queue_
    std::unique_ptr<TaskType> cur_task;
    std::shared_ptr<ProcessFunc> cur_func;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      cur_task = std::move(queue_.front().task); 
      cur_func = std::move(queue_.front().process_func);
      queue_.pop();
    }
    // start to process func
    (*cur_func)(std::move(cur_task)); 
    empty_.notify(); 
  }  
}

} // namespace blaze

#endif  // BLAZE_SCHEDULER_SIMPLE_SCHEDULER_H_
