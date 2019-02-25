// Batch Scheduler Retrier 
//
#pragma once

#include "blaze/batching/batch_scheduler.h"
#include "blaze/batching/env.h"

namespace blaze {
namespace batching {

template <typename TaskType>
class BatchSchedulerRetrier : public BatchScheduler<TaskType> {
 public:
  struct Options {
    /// The maximum amount of time to spend retrying 'wrapped_->Schedule()'
    /// calls, in microseconds.
    int64_t max_time_micros = 10 * 1000;

    /// The amount of time to pause between retry attempts
    int64_t retry_delay_micros = 100;
  };
  static bool Create(const Options& options,
                     std::unique_ptr<BatchScheduler<TaskType>> wrapped,
                     std::unique_ptr<BatchSchedulerRetrier<TaskType>>* result);
  ~BatchSchedulerRetrier() override = default;

  Status Schedule(std::unique_ptr<TaskType>* task) override;
  size_t NumEnqueuedTasks() const override;
  size_t SchedulingCapacity() const override;

 private:
  BatchSchedulerRetrier(const Options& options,
                        std::unique_ptr<BatchScheduler<TaskType>> wrapped);

  const Options options_;
  std::unique_ptr<BatchScheduler<TaskType>> wrapped_;
};

template <typename TaskType>
bool BatchSchedulerRetrier<TaskType>::Create(const Options& options,
                                             std::unique_ptr<BatchScheduler<TaskType>> wrapped,
                                             std::unique_ptr<BatchSchedulerRetrier<TaskType>>* result) {
  if (options.max_time_micros < 0) {
    LOG_ERROR("max_time_micros must be non-negative; was %d", options.max_time_micros);
    return false;
  }
  if (options.retry_delay_micros < 0) {
    LOG_ERROR("retry_delay_micros must be non-negative; was %d", options.retry_delay_micros);
    return false;
  }
  result->reset(new BatchSchedulerRetrier(options, std::move(wrapped)));
  return true;
}

template <typename TaskType>
Status BatchSchedulerRetrier<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  Status status;
  const uint64_t start_time_micros = Env::NowMicros();
  for (;;) {
    status = wrapped_->Schedule(task);
    if (status != kUnavailable) {
      break;
    }
    if ((Env::NowMicros() + options_.retry_delay_micros) - start_time_micros >=
        options_.max_time_micros) {
      break;
    }
    Env::SleepForMicroseconds(options_.retry_delay_micros);
  }
  return status;
}

template <typename TaskType>
size_t BatchSchedulerRetrier<TaskType>::NumEnqueuedTasks() const {
  return wrapped_->NumEnqueuedTasks();
}

template <typename TaskType>
size_t BatchSchedulerRetrier<TaskType>::SchedulingCapacity() const {
  return wrapped_->SchedulingCapacity();
}

template <typename TaskType>
BatchSchedulerRetrier<TaskType>::BatchSchedulerRetrier(
    const Options& options, std::unique_ptr<BatchScheduler<TaskType>> wrapped)
  : options_(options), wrapped_(wrapped) { }

}  // namespace batching
}  // namespace blaze
