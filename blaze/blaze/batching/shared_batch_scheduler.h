// Shared Batch Scheduler 
//
#pragma once

#include <deque>
#include <string>
#include <list>

#include "blaze/scheduler/scheduler.h"
#include "blaze/common/common_defines.h"
#include "blaze/common/exception.h"
#include "blaze/batching/batch_scheduler.h"
#include "blaze/batching/periodic_function.h"

namespace blaze {
namespace batching {

/// Pre declare
namespace internal {
template <typename TaskType> class Queue;
}  // namespace internal

/// A batch scheduler for server instances that service mutiple request type
/// (eg. mutiple models, or mutiple versions of a model served concurrently)
template <typename TaskType>
class SharedBatchScheduler : public std::enable_shared_from_this<SharedBatchScheduler<TaskType>>,
    public Scheduler {
 public:
  struct Options {
    /// The name to use for the pool of batch threads
    std::string thread_pool_name = {"batch_threads"};

    /// The number of threads to use to process batches
    /// Must be >= 1, and should be tuned carefully.
    int num_batch_threads = Env::NumSchedulableCPUs();
  };
  /// Ownership is shard between the caller of Create() and any queues created
  /// via AddQueue.
  static bool Create(const Options& options, std::shared_ptr<SharedBatchScheduler<TaskType>>* scheduler);

  ~SharedBatchScheduler();

  /// Adds a queue to which tasks may be submitted. the returned queue
  /// implements the BatchScheduler API. each queue has its own set of
  /// scheduling options and its own callback to process batches of tasks
  /// submitted to the queue
  struct QueueOptions {
    /// The maximum size of each batch
    int max_batch_size = 1000;

    /// bound queue latency
    int64_t batch_timeout_micros = 0;

    /// maximum allowable number of enqueued tasks in terms of batches.
    /// if this limit is reached, Schedule will return an Unavailable error.
    int64_t max_enqueued_batches = 10;
  };
  /// Add Queue
  bool AddQueue(const QueueOptions& options,
                std::function<void(std::unique_ptr<Batch<TaskType>>)> process_batch_callback,
                std::unique_ptr<BatchScheduler<TaskType>>* queue);

 private:
  explicit SharedBatchScheduler(const Options& options);

  /// The code executed in 'batch_threads_*', Obtains a batch to process from
  /// the queue pointed to by 'next_queue_to_schedule_', and processed it.
  void ThreadLogic();

  const Options options_;
  mutex mu_;
  condition_variable schedulable_batch_cv_;

  /// A list of queues.
  using QueueList = std::list<std::unique_ptr<internal::Queue<TaskType>>>;
  /// All "active" queue, i.e. ones that either:
  ///  - have not been removed, or
  ///  - have been removed but are not yet empty
  QueueList queues_;
  // An iterator ove queues_
  typename QueueList::iterator next_queue_to_schedule_;

  /// Threads that process batches obtained from the queues
  std::vector<std::unique_ptr<PeriodicFunction>> batch_threads_;
};

namespace internal {

/// A task queue for SharedBatchScheduler. Accepts tasks and accumulates them
/// into batches, and dispenses those batches to be processed via a "pull"
/// interface. The queue's behavior is governed by maximum batch size, timeout
/// and maximum queue length parameters
template <typename TaskType>
class Queue {
 public:
  using ProcessBatchCallback =
      std::function<void(std::unique_ptr<Batch<TaskType>>)>;
  using SchedulableBatchCallback = std::function<void()>;
  Queue(const typename SharedBatchScheduler<TaskType>::QueueOptions& options,
        ProcessBatchCallback process_batch_callback,
        SchedulableBatchCallback schdulable_batch_callback);
  ~Queue();

  /// Submit a task to the queue, with the same semantics as
  /// BatchScheduler::Schedule
  Status Schedule(std::unique_ptr<TaskType>* task);

  /// Returns the number of enqueued tasks, with the same semantics as
  /// BatchSchedule::NumEnqueuedTasks()
  size_t NumEnqueuedTasks() const;

  /// Returns the queue capacity, with the same semantics as
  /// BatchScheduler::SchedulingCapacity()
  size_t SchedulingCapacity() const;

  /// Called by a thread that is ready to process a batch
  std::unique_ptr<Batch<TaskType>> ScheduleBatch();

  /// Processes a batch that has been returned earlier by ScheduleBatch()
  void ProcessBatch(std::unique_ptr<Batch<TaskType>> batch);

  /// Determines whether the queue is empty
  bool IsEmpty() const;

  /// Marks the queue closed, and waits until it is empty
  void CloseAndWaitUntilEmpty();

  /// closed 
  bool closed() const {
    mutex_lock l(mu_);
    return closed_;
  }

 private:
  /// Same as IsEmpty, but assume the caller already holds a lock on mu_
  bool IsEmptyInternal() const;

  /// Closes the open batch residing at the back of 'batches_', and inserts a
  /// fresh open batch behind it
  void StartNewBatch();

  /// Determines whether the open batch residing at the back of 'batches_' is
  /// currently schedulable.
  bool IsOpenBatchSchedulable();

  /// queue options
  const typename SharedBatchScheduler<TaskType>::QueueOptions options_;
  /// batch callback
  ProcessBatchCallback process_batch_callback_;
  /// a callback invoked to notify the scheduler that a new batch has become
  /// schedulable
  SchedulableBatchCallback schedulable_batch_callback_;

  mutable mutex mu_;
  /// Whether this queue can accept new tasks. This variable is monotonic: it
  /// starts as false, and then at some point gets set to true and remains true
  /// for the duration of this object's life.
  bool closed_ = false;

  /// The enqueued batches
  std::deque<std::unique_ptr<Batch<TaskType>>> batches_;
  /// The time at which the first task was added to the open (back-most) batch
  /// in 'batches_'. Valid iff that batch contains at least one task
  int64_t open_batch_start_time_micros_;

  /// Whether this queue contains a batch that is eligible to be scheduled
  bool schedulable_batch_ = false;

  /// The number of batches currently being processed by batch threads
  int num_batches_being_processed_ = 0;

  /// Used by CloseAndWaitUntilEmpty() to wait until the queue is empty.
  Notification* empty_notification_ = nullptr;
};

/// A RAII-style object that points to a Queue and implements
/// the BatchScheduler API. To be handed out to clients who call AddQueue()
template <typename TaskType>
class QueueHandle : public BatchScheduler<TaskType> {
 public:
  QueueHandle(std::shared_ptr<SharedBatchScheduler<TaskType>> scheduler,
              Queue<TaskType>* queue);
  ~QueueHandle() override;

  Status Schedule(std::unique_ptr<TaskType>* task) override;
  size_t NumEnqueuedTasks() const override;
  size_t SchedulingCapacity() const override;

 private:
  std::shared_ptr<SharedBatchScheduler<TaskType>> scheduler_;
  Queue<TaskType>* queue_;
};

}  // namepsace internal

template <typename TaskType>
bool SharedBatchScheduler<TaskType>::Create(const Options& options,
                                            std::shared_ptr<SharedBatchScheduler<TaskType>>* scheduler) {
  if (options.num_batch_threads < 1) {
    BLAZE_CONDITION_THROW("num_batch_threads must be positive; was ",
                          options.num_batch_threads);
    return false;
  }
  scheduler->reset(new SharedBatchScheduler<TaskType>(options));
  return true;
}

template <typename TaskType>
SharedBatchScheduler<TaskType>::~SharedBatchScheduler() {
  /// Wait util the batch threads finish clearing out and deleting the closed
  /// queues.
  for (;;) {
    LOG_INFO("Deleting SharedBatchScheduler, SharedScheduler Exiting!!!");
    mutex_lock l(mu_);
    if (queues_.empty()) {
      break;
    }
    const int64_t kSleepTimeMicros = 100;
    Env::SleepForMicroseconds(kSleepTimeMicros);
  }
  batch_threads_.clear();
}
  
template <typename TaskType>
bool SharedBatchScheduler<TaskType>::AddQueue(const QueueOptions& options,
                                              std::function<void(std::unique_ptr<Batch<TaskType>>)> process_batch_callback,
                                              std::unique_ptr<BatchScheduler<TaskType>>* queue) {
  if (options.max_batch_size < 1) {
    BLAZE_CONDITION_THROW("max_batch_size must be positive; was ", options.max_batch_size);
    return false;
  }
  if (options.batch_timeout_micros < 0) {
    BLAZE_CONDITION_THROW("batch_timeout_micros must be non-negative; was ", options.batch_timeout_micros);
    return false;
  }
  if (options.max_enqueued_batches < 0) {
    BLAZE_CONDITION_THROW("batch_timeout_micros must be non-negative; was ", options.max_enqueued_batches);
    return false;
  }
  auto schedulable_batch_callback = [this] {
    mutex_lock l(mu_);
    schedulable_batch_cv_.notify_one();
  };
  auto internal_queue = std::unique_ptr<internal::Queue<TaskType>>(new internal::Queue<TaskType>(
          options, process_batch_callback, schedulable_batch_callback));
  auto handle = std::unique_ptr<BatchScheduler<TaskType>>(
        new internal::QueueHandle<TaskType>(this->shared_from_this(), internal_queue.get()));
  {
    mutex_lock l(mu_);
    queues_.push_back(std::move(internal_queue));
    if (next_queue_to_schedule_ == queues_.end()) {
      next_queue_to_schedule_ = queues_.begin();
    }
  }
  
  *queue = std::move(handle);
  return true;
}

template <typename TaskType>
SharedBatchScheduler<TaskType>::SharedBatchScheduler(const Options& options) :
    options_(options), next_queue_to_schedule_(queues_.end()) {
  /// Kick off the batch threads
  PeriodicFunction::Options periodic_fn_options;
  periodic_fn_options.thread_name_prefix = options.thread_pool_name + std::string("_");
  for (int i = 0; i < options.num_batch_threads; ++i) {
    std::unique_ptr<PeriodicFunction> thread(new PeriodicFunction(
            [this] { this->ThreadLogic(); },
            0 /* function invocation interval time */, periodic_fn_options));
    batch_threads_.push_back(std::move(thread));
  }
}

template <typename TaskType>
void SharedBatchScheduler<TaskType>::ThreadLogic() {
  /// A batch to process next (or nullptr if no work to do).
  std::unique_ptr<Batch<TaskType>> batch_to_process;
  /// The queue with which "batch_to_process" is associated
  internal::Queue<TaskType>* queue_for_batch = nullptr;
  {
    mutex_lock l(mu_);
    
    const int num_queues = queues_.size();
    for (int num_queues_tried = 0; batch_to_process == nullptr && num_queues_tried < num_queues;
         ++num_queues_tried) {
      /// If a closed queue responds to ScheduleBatch() with nullptr, the queue
      /// will never yield any further batches so we can drop it. To avoid race,
      /// we tack a snapshot of the queue's closedness state *before* calling
      /// ScheduleBatch().
      const bool queue_closed = (*next_queue_to_schedule_)->closed();

      /// Ask '*next_queue_to_schedule_' if it wants us to process a batch
      batch_to_process = (*next_queue_to_schedule_)->ScheduleBatch();
      if (batch_to_process != nullptr) {
        queue_for_batch = next_queue_to_schedule_->get();
      }

      /// Advance 'next_queue_to_schedule'
      if (queue_closed && (*next_queue_to_schedule_)->IsEmpty() &&
          batch_to_process == nullptr) {
        /// We have encountered a closed queue with no work to do, drop it.
        next_queue_to_schedule_ = queues_.erase(next_queue_to_schedule_);
      } else {
        ++next_queue_to_schedule_;
      }
      if (next_queue_to_schedule_ == queues_.end() && !queues_.empty()) {
        /// We have hit the end. Wrap to the first queue
        next_queue_to_schedule_ = queues_.begin();
      }
    }

    if (batch_to_process == nullptr) {
      /// We could'not find any work to do. Wait until a new batch becomes
      /// scheduable, or some time has elapsed, before checking again.
      const int64_t kTimeoutMillis = 1;
      WaitForMilliseconds(&l, &schedulable_batch_cv_, kTimeoutMillis);
      return;
    }
  }
  queue_for_batch->ProcessBatch(std::move(batch_to_process));
}

namespace internal {

template <typename TaskType>
Queue<TaskType>::Queue(const typename SharedBatchScheduler<TaskType>::QueueOptions& options,
                       ProcessBatchCallback process_batch_callback,
                       SchedulableBatchCallback schedulable_batch_callback) :
    options_(options),
    process_batch_callback_(process_batch_callback),
    schedulable_batch_callback_(schedulable_batch_callback) {
  /// Create an initial, open batch
  batches_.emplace_back(new Batch<TaskType>);
}

template <typename TaskType>
Queue<TaskType>::~Queue() {
  mutex_lock l(mu_);
  /// Close the (empty) open batch, so its destructor doesn't block.
  batches_.back()->Close();
}

template <typename TaskType>
Status Queue<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  bool notify_of_schedulable_batch = false;
  {
    mutex_lock l(mu_);

    if (batches_.back()->size() + (*task)->size() > options_.max_batch_size) {
      if (batches_.size() >= options_.max_enqueued_batches) {
        return kUnavailable;
      }
      StartNewBatch();
    }
    if (batches_.back()->empty()) {
      open_batch_start_time_micros_ = Env::NowMicros();
    }
    batches_.back()->AddTask(std::move(*task));

    if (!schedulable_batch_) {
      if (batches_.size() > 1 || IsOpenBatchSchedulable()) {
        schedulable_batch_ = true;
        notify_of_schedulable_batch = true;
      }
    }
  }
  if (notify_of_schedulable_batch) {
    schedulable_batch_callback_();
  }
  return kOk;
}

template <typename TaskType>
size_t Queue<TaskType>::NumEnqueuedTasks() const {
  mutex_lock l(mu_);
  size_t num_enqueued_tasks = 0;
  for (const auto& batch : batches_) {
    num_enqueued_tasks += batch->num_tasks();
  }
  return num_enqueued_tasks;
}

template <typename TaskType>
size_t Queue<TaskType>::SchedulingCapacity() const {
  mutex_lock l(mu_);
  const int num_new_batches_schedulable =
      options_.max_enqueued_batches - batches_.size();
  const int open_batch_capacity =
      options_.max_batch_size - batches_.back()->size();
  return (num_new_batches_schedulable * options_.max_batch_size) +
      open_batch_capacity;
}

template <typename TaskType>
std::unique_ptr<Batch<TaskType>> Queue<TaskType>::ScheduleBatch() {
  /// Schedule the batch
  std::unique_ptr<Batch<TaskType>> batch_to_schedule;

  {
    mutex_lock l(mu_);

    /// Consider closeing the open batch at this time, to schedule it
    if (batches_.size() == 1 && IsOpenBatchSchedulable()) {
      StartNewBatch();
    }
    if (batches_.size() >= 2) {
      ++num_batches_being_processed_;
      batch_to_schedule = std::move(batches_.front());
      batches_.pop_front();
    } else {
      schedulable_batch_ = false;
    }
  }
  return batch_to_schedule;
}

template <typename TaskType>
void Queue<TaskType>::ProcessBatch(std::unique_ptr<Batch<TaskType>> batch) {
  process_batch_callback_(std::move(batch));
  {
    mutex_lock l(mu_);
    --num_batches_being_processed_;
    if (empty_notification_ != nullptr && IsEmptyInternal()) {
      empty_notification_->Notify();
    }
  }
}

template <typename TaskType>
bool Queue<TaskType>::IsEmpty() const {
  mutex_lock l(mu_);
  return IsEmptyInternal();
}

template <typename TaskType>
void Queue<TaskType>::CloseAndWaitUntilEmpty() {
  Notification empty;
  {
    mutex_lock l(mu_);
    closed_ = true;
    if (IsEmptyInternal()) {
      empty.Notify();
    } else {
      /// Arrange for ProcessBatch() to notify
      empty_notification_ = &empty;
    }
  }
  empty.WaitForNotification();
}

template <typename TaskType>
bool Queue<TaskType>::IsEmptyInternal() const {
  return num_batches_being_processed_ == 0 && batches_.size() == 1 &&
      batches_.back()->empty();
}

template <typename TaskType>
void Queue<TaskType>::StartNewBatch() {
  batches_.back()->Close();
  batches_.emplace_back(new Batch<TaskType>);
}

template <typename TaskType>
bool Queue<TaskType>::IsOpenBatchSchedulable() {
  Batch<TaskType>* open_batch = batches_.back().get();
  if (open_batch->empty()) {
    return false;
  }
  return closed_ || open_batch->size() >= options_.max_batch_size ||
      Env::NowMicros() >= open_batch_start_time_micros_ + options_.batch_timeout_micros;
}

/// Interface of SharedBatchScheduler like as BatchScheduler.
template <typename TaskType>
QueueHandle<TaskType>::QueueHandle(std::shared_ptr<SharedBatchScheduler<TaskType>> scheduler,
                                   Queue<TaskType>* queue)
  : scheduler_(scheduler), queue_(queue) { }

template <typename TaskType>
QueueHandle<TaskType>::~QueueHandle() {
  queue_->CloseAndWaitUntilEmpty();
}

template <typename TaskType>
Status QueueHandle<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  return queue_->Schedule(task);
}

template <typename TaskType>
size_t QueueHandle<TaskType>::NumEnqueuedTasks() const {
  return queue_->NumEnqueuedTasks();
}

template <typename TaskType>
size_t QueueHandle<TaskType>::SchedulingCapacity() const {
  return queue_->SchedulingCapacity();
}

}  // namespace internal

}  // namespace batching
}  // namespace blaze
