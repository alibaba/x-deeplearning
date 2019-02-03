/*!
 * \file cuda_event.h
 * \brief The cuda event.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>

#include "blaze/common/context.h"
#include "blaze/common/event.h"
#include "blaze/common/log.h"

namespace blaze {

struct CudaEventWrapper {
  explicit CudaEventWrapper(const DeviceOption& option) :
      cuda_stream_(nullptr),
      cuda_gpu_id_(option.device_id()),
      status_(EventStatus::kEventInitialized) {
    CHECK(option.device_type() == kCUDA, "Expect device_type equals kCUDA");
    CUDADeviceGuard g(cuda_gpu_id_);
    CUDA_CHECK(cudaEventCreate(&cuda_event_, cudaEventDefault | cudaEventDisableTiming));
  }

  ~CudaEventWrapper() {
    CUDADeviceGuard g(cuda_gpu_id_);
    cudaEventDestroy(cuda_event_);
  }

  cudaEvent_t cuda_event_;
  cudaStream_t cuda_stream_;
  int cuda_gpu_id_;

  std::atomic<int> status_;
  std::mutex mutex_recorded_;
  std::condition_variable cv_recorded_;
  std::string err_msg_;
};

void EventCreateCUDA(const DeviceOption& option, Event* event);
void EventRecordCUDA(Event* event, const void* context, const char* err_msg);
void EventFinishCUDA(const Event* event);
void EventWaitCUDACUDA(const Event* event, void* context);
void EventWaitCPUCUDA(const Event* event, void* context);
void EventWaitCUDACPU(const Event* event, void* context);
EventStatus EventQueryCUDA(const Event* event);
const std::string& EventErrorMessageCUDA(const Event* event);
void EventSetFinishedCUDA(const Event* event, const char* err_msg);
void EventResetCUDA(Event* event);

}  // namespace blaze

