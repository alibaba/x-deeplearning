// Basic Batch Scheduler 
//
#pragma once

#include "blaze/batching/shared_batch_scheduler.h"

namespace blaze {
namespace batching {

template <typename TaskType>
class BasicBatchScheduler : public BatchScheduler<TaskType> {
 public:
  struct Options {
    /// Maximum batch size
    int max_batch_size = 1000;

    /// Batch timeout micros
    int64_t batch_timeout_micros = 0;

    /// Name of thread pool
    std::string thread_pool_name = { "batch_threads" };

    /// The number of threads to process batch
    int num_batch_threads = Env::NumSchedulableCPUs();

    /// Max enqueued batches
    int max_enqueued_batches = 10;
  };
  static bool Create(const Options& options,
                     std::function<void(std::unique_ptr<Batch<TaskType>>)> process_batch_callback,
                     std::unique_ptr<BasicBatchScheduler>* scheduler);
  ~BasicBatchScheduler() override = default;

  Status Schedule(std::unique_ptr<TaskType>* task) override;
  size_t NumEnqueuedTasks() const override;
  size_t SchedulingCapacity() const override;

 private:
  explicit BasicBatchScheduler(std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue);

  std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue_;
};

template <typename TaskType>
bool BasicBatchScheduler<TaskType>::Create(const Options& options,
                                           std::function<void(std::unique_ptr<Batch<TaskType>>)> process_batch_callback,
                                           std::unique_ptr<BasicBatchScheduler>* scheduler) {
  typename SharedBatchScheduler<TaskType>::Options shared_scheduler_options;
  shared_scheduler_options.thread_pool_name = options.thread_pool_name;
  shared_scheduler_options.num_batch_threads = options.num_batch_threads;
  
  std::shared_ptr<SharedBatchScheduler<TaskType>> shared_scheduler;
  if (!SharedBatchScheduler<TaskType>::Create(shared_scheduler_options, &shared_scheduler)) {
    return false;
  }

  typename SharedBatchScheduler<TaskType>::QueueOptions shared_scheduler_queue_options;
  shared_scheduler_queue_options.max_batch_size = options.max_batch_size;
  shared_scheduler_queue_options.batch_timeout_micros = options.batch_timeout_micros;
  shared_scheduler_queue_options.max_enqueued_batches = options.max_enqueued_batches;

  std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue;
  if (!shared_scheduler->AddQueue(shared_scheduler_queue_options,
                                  process_batch_callback,
                                  &shared_scheduler_queue)) {
    return false;
  }
  scheduler->reset(new BasicBatchScheduler<TaskType>(std::move(shared_scheduler_queue)));
  return true;
}

template <typename TaskType>
Status BasicBatchScheduler<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  return shared_scheduler_queue_->Schedule(task);
}

template <typename TaskType>
size_t BasicBatchScheduler<TaskType>::NumEnqueuedTasks() const {
  return shared_scheduler_queue_->NumEnqueuedTasks();
}

template <typename TaskType>
size_t BasicBatchScheduler<TaskType>::SchedulingCapacity() const {
  return shared_scheduler_queue_->SchedulingCapacity();
}

template <typename TaskType>
BasicBatchScheduler<TaskType>::BasicBatchScheduler(
    std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue)
  : shared_scheduler_queue_(std::move(shared_scheduler_queue)) { }

}  // namespace batching
}  // namespace blaze
