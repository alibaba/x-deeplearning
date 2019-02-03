/*
 * \file scheduler_manager.h 
 * \brief The scheduler manager for inference of multi models 
 */
#ifndef BLAZE_SCHEDULER_SCHEDULER_MANAGER_H_
#define BLAZE_SCHEDULER_SCHEDULER_MANAGER_H_

#include <unordered_map>
#include <memory>
#include <mutex>
#include "blaze/scheduler/scheduler.h"
#include "blaze/batching/shared_batch_scheduler.h"
#include "blaze/scheduler/simple_scheduler.h"
#include "blaze/proto/blaze.pb.h"

namespace blaze {

#define RET_IF_FAILED(func, stmt) \
  do {                            \
    if (!(func)) {                \
      LOG_ERROR(stmt);            \
      return false;               \
    }                             \
  } while (0)

/**
 * NOTE: For each device, there is only one scheduler attached with it
 */
template <typename TaskType> 
class SchedulerManager {
 public:
  struct Options {
    // number of concurrent threads
    int num_threads_for_cpu = 4;
    int num_threads_for_cuda = 4;
    int num_threads_for_pipe = 2;
    // queue capacity for non-batching queue  
    int queue_capacity = 500;

    // enable batching or not
    bool enable_batching = false;
    // number of batch threads
    int num_batch_threads = 2;
    // max batch size 
    int max_batch_size = 100; 
    // batch timeout (microseconds)
    int batch_timeout_micros = 100;
    // maximum allowable number of enqueued tasks in terms of batches
    // if this limit is reached, Schedule will return an Unavailable error
    int max_enqueued_batches = 500;
  };

  // Singleton instance
  static std::shared_ptr<SchedulerManager<TaskType>> Instance();
  
  ~SchedulerManager() {
    LOG_DEBUG("Delete scheduler manager");
  }

  void Destroy() {
    has_init_ = false;
    net_def_batched_queue_.clear();
    schedulers_.clear();
    cuda_schedulers_.clear();
  }

  // Init schedulers for all devices, thread safe 
  bool Init(const Options& options);

  inline Scheduler* GetScheduler(const DeviceOption& device_option) const {
    if (device_option.device_type() == kCUDA && !device_option.is_pipe()) {
      // for cuda
      auto it = cuda_schedulers_.find(device_option.device_type());
      if (it != cuda_schedulers_.end()) {
        auto inner_it = it->second.find(device_option.device_id());
        if (inner_it != it->second.end()) {
          return inner_it->second.get();
        }
      } 
    } else {
      // for normal cpu/pipe
      auto it = schedulers_.find(device_option.device_type());
      if (it != schedulers_.end()) {
        auto inner_it = it->second.find(device_option.device_id());
        if (inner_it != it->second.end()) {
          return inner_it->second.get();
        }
      }  
    }
    return nullptr;
  }

  // each net_def is mapped to a BatchedQueue
  batching::BatchScheduler<TaskType>* GetBatchedQueue(const NetDef* net_def,
      std::function<void(std::unique_ptr<batching::Batch<TaskType>>)> process_cb,
      batching::SharedBatchScheduler<TaskType>* batch_scheduler) {
    auto it = net_def_batched_queue_.find(net_def);
    if (it != net_def_batched_queue_.end()) {
      return it->second.get();
    }
    std::unique_lock<std::mutex> lock(mutex_);
    it = net_def_batched_queue_.find(net_def);
    if (it != net_def_batched_queue_.end()) {
      return it->second.get();
    }
    typename batching::SharedBatchScheduler<TaskType>::QueueOptions queue_options;
    queue_options.max_batch_size = options_.max_batch_size;
    queue_options.batch_timeout_micros = options_.batch_timeout_micros;
    queue_options.max_enqueued_batches = options_.max_enqueued_batches;
    std::unique_ptr<batching::BatchScheduler<TaskType>> queue; 
    batch_scheduler->AddQueue(queue_options, process_cb, &queue);
    net_def_batched_queue_[net_def] = std::move(queue);
    return net_def_batched_queue_[net_def].get();
  }

  // when a workspace is destroyed
  // its corresponding batchedQueue should be released as well
  void ReleaseBatchedQueue(const NetDef* net_def) {
    std::unique_lock<std::mutex> lock(mutex_);
    net_def_batched_queue_.erase(net_def); 
  }

 private:
  using DeviceIdScheduler = std::unordered_map<int, std::shared_ptr<Scheduler>>;
  using DeviceOptionScheduler = std::unordered_map<int, DeviceIdScheduler>;
  
  bool InitOneScheduler(int num_threads, int queue_capacity,
      int device_count, int device_type, DeviceOptionScheduler* schedulers);

  SchedulerManager() : has_init_(false) {
  }
  DISABLE_COPY_AND_ASSIGN(SchedulerManager);

  // <device_type, <device id, scheduler>>
  DeviceOptionScheduler schedulers_;
  DeviceOptionScheduler cuda_schedulers_;

  std::unordered_map<const NetDef*,
    std::unique_ptr<batching::BatchScheduler<TaskType>>> net_def_batched_queue_;
  std::mutex mutex_;

  bool has_init_;
  Options options_;
}; 

template <typename TaskType>
bool SchedulerManager<TaskType>::InitOneScheduler(int num_threads, int queue_capacity,
    int device_count, int device_type, DeviceOptionScheduler* schedulers) {
  typename SimpleScheduler<TaskType>::Options options;
  options.num_threads = num_threads;
  options.queue_capacity = queue_capacity;
  for (int i = 0; i < device_count; ++i) {
    std::shared_ptr<SimpleScheduler<TaskType>> scheduler;
    RET_IF_FAILED(SimpleScheduler<TaskType>::Create(options, &scheduler),
        "Create simple scheduler failed");
    (*schedulers)[device_type][i] = std::move(scheduler); 
  }
  return true;
}


template <typename TaskType> 
bool SchedulerManager<TaskType>::Init(const Options& options) {
  if (has_init_) {
    LOG_ERROR("Reinit SchedulerManager");
    return false;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  if (has_init_) {
    LOG_ERROR("Reinit SchedulerManager");
    return false;
  }
  // read options to build schedulers
  options_ = options;
  
#ifdef USE_CUDA
  int gpu_count = 0;  
  CUDA_CHECK(cudaGetDeviceCount(&gpu_count)); 
  // for cuda
  if (options.enable_batching) {
    typename batching::SharedBatchScheduler<TaskType>::Options cuda_options;
    cuda_options.num_batch_threads = options.num_batch_threads;
    for (int i = 0; i < gpu_count; ++i) {
      std::shared_ptr<batching::SharedBatchScheduler<TaskType>> scheduler;
      RET_IF_FAILED(batching::SharedBatchScheduler<TaskType>::Create(
          cuda_options, &scheduler), "Create batch scheduler for cuda failed");
      cuda_schedulers_[kCUDA][i] = std::move(scheduler); 
    } 
  } else {
    RET_IF_FAILED(InitOneScheduler(options.num_threads_for_cuda,
          options.queue_capacity, gpu_count, kCUDA, &cuda_schedulers_),
        "Create simple scheduler for cuda failed");
  }

  // for pipe
  RET_IF_FAILED(InitOneScheduler(options.num_threads_for_pipe,
        options.queue_capacity, gpu_count, kCUDA, &schedulers_),
      "Create simple scheduler for cuda failed");
#endif

  // for cpu 
  RET_IF_FAILED(InitOneScheduler(options.num_threads_for_cpu,
        options.queue_capacity, 1, kCPU, &schedulers_),
      "Create simple scheduler for cpu failed");

  has_init_ = true;

  return true;
}

template <typename TaskType> 
std::shared_ptr<SchedulerManager<TaskType>> SchedulerManager<TaskType>::Instance() {
  static std::shared_ptr<SchedulerManager<TaskType>> scheduler_manager(
      new SchedulerManager<TaskType>()); 
  return scheduler_manager; 
}

} // namespace blaze

#endif  // BLAZE_SCHEDULER_SCHEDULER_MANAGER_H_
