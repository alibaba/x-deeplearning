// Abstract Batch Scheduler 
//
#pragma once

#include <vector>

#include "blaze/batching/mutex.h"
#include "blaze/batching/notification.h"

namespace blaze {
namespace batching {

/// The abstract superclass for a unit of work to be done as part of a batch
///
/// An implemention subclass contains:
///  (a) input data
///  (b) a thread-safe completion signal (eg. Notification)
///  (c) a place to store the outcome(success or some error)
///  (d) a place to store the output data
class BatchTask {
 public:
  virtual ~BatchTask() = default;

  /// Returns the size of the task, in terms of how much it contributes to the
  /// size of a batch. (A batch's size is the sum of its task sizes).
  virtual size_t size() const = 0;
};

/// A thread-safe collection of BatchTasks, to be executed together in some
/// fasion.
///
/// The parameter TaskType must be a subclass of BatchTask
template <typename TaskType>
class Batch {
 public:
  Batch() = default;
  ~Batch();  /// Blocks until the batch is close.

  /// Append 'task' to the batch. After calling AddTask(), the newly-added task
  /// can be accessed via task(num_tacks() - 1) or mutable_task(num_tasks() - 1).
  /// Dies if the batch is closed.
  void AddTask(std::unique_ptr<TaskType> task);

  /// Removes the most recently added tasks. Returns nullptr if batch is empty
  std::unique_ptr<TaskType> RemoveTask();

  /// Returns the number of tasks in the batch
  int num_tasks() const;

  /// Returns true iff the batch contains 0 tasks
  bool empty() const;

  /// Returns a reference to the ith task (in terms of inertion order)
  const TaskType& task(int i) const;

  /// Returns a pointer to the ith task (in terms of insertion order)
  TaskType* mutable_task(const int);

  /// Returns the sum of the task sizes
  size_t size() const;

  /// Returns ture iff the batch is currently closed
  bool IsClosed() const;

  /// Blocks untils the batch is closed
  void WaitUntilClosed() const;

  /// Marks the batch are closed, Dir if called more than once
  void Close();

 private:
  mutable mutex mu_;

  /// The tasks in batch
  std::vector<std::unique_ptr<TaskType>> tasks_;

  /// The sum size of the tasks in "task_"
  size_t size_ = 0;

  /// Whether the batch has been closed
  Notification closed_;
};

/// Status of Schedule
enum Status {
  kOk = 0,
  kUnavailable = 1,
  kError = 2,
};

/// An abstract batch scheduler class. Collects individial tasks into batches.
/// and process each batch on a pool of "batch threads" that it manages. The
/// actual logic for processing a batch is accomplished via a callback.
///
/// Type parameter TaskType must be a subclass of BatchTask
template <typename TaskType>
class BatchScheduler {
 public:
  virtual ~BatchScheduler() = default;

  /// Submit a task to be processed as part of a batch.
  ///
  /// Ownship of task is transfered to the callee iff the method return true, in
  /// that case, the task is left as nullptr, Otherwise, task is left as-is.
  ///
  /// If no batching processing capacity is available to process this task,
  /// return Unavailable error code
  ///
  /// Other problems, return Error code.
  virtual Status Schedule(std::unique_ptr<TaskType>* task) = 0;

  /// Returns the number of tasks that have been scheduled.
  virtual size_t NumEnqueuedTasks() const = 0;

  /// Returns a guaranteed number of size 1 tasks that can be Schedule()d
  /// without getting an UNAVAILABLE error
  virtual size_t SchedulingCapacity() const = 0;
};

/// Implemation
template <typename TaskType>
Batch<TaskType>::~Batch() {
  WaitUntilClosed();
}

template <typename TaskType>
void Batch<TaskType>::AddTask(std::unique_ptr<TaskType> task) {
  {
    mutex_lock l(mu_);
    size_ += task->size();
    tasks_.push_back(std::move(task));
  }
}

template <typename TaskType>
std::unique_ptr<TaskType> Batch<TaskType>::RemoveTask() {
  mutex_lock l(mu_);
  if (tasks_.empty()) {
    return nullptr;
  }
  std::unique_ptr<TaskType> task = std::move(tasks_.back());
  tasks_.pop_back();
  return task;
}

template <typename TaskType>
int Batch<TaskType>::num_tasks() const {
  mutex_lock l(mu_);
  return tasks_.size();
}

template <typename TaskType>
bool Batch<TaskType>::empty() const {
  mutex_lock l(mu_);
  return tasks_.empty();
}

template <typename TaskType>
const TaskType& Batch<TaskType>::task(int i) const {
  {
    mutex_lock l(mu_);
    return *tasks_[i].get();
  }
}

template <typename TaskType>
TaskType* Batch<TaskType>::mutable_task(int i) {
  {
    mutex_lock l(mu_);
    return tasks_[i].get();
  }
}

template <typename TaskType>
size_t Batch<TaskType>::size() const {
  mutex_lock l(mu_);
  return size_;
}

template <typename TaskType>
bool Batch<TaskType>::IsClosed() const {
  return const_cast<Notification*>(&closed_)->HasBeenNotified();
}

template <typename TaskType>
void Batch<TaskType>::WaitUntilClosed() const {
  const_cast<Notification*>(&closed_)->WaitForNotification();
}

template <typename TaskType>
void Batch<TaskType>::Close() {
  closed_.Notify();
}

}  // namespace batching
}  // namespace blaze
