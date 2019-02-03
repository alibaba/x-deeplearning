/*!
 * \file cuda_event.cc
 * \brief The cuda event.
 */
#ifdef USE_CUDA

#include "blaze/common/cuda_event.h"

namespace {
const std::string kNoError = "No error";
}  // namespace

namespace blaze {

void EventCreateCUDA(const DeviceOption& option, Event* event) {
  event->event_ = std::make_shared<CudaEventWrapper>(option);
}

void EventRecordCUDA(Event* event, const void* context, const char* err_msg) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
    
    // Possible state changes:
    //   Initialized -> Scheduled/Failed
    //   Scheduled -> Success/Failed
    //   Success/Failed -> terminal
    CHECK(wrapper->status_ == EventStatus::kEventInitialized,
          "Calling Record mutiple times");

    if (!err_msg) {
      CUDA_CHECK(cudaEventRecord(wrapper->cuda_event_,
                                 static_cast<const CUDAContext*>(context)->cuda_stream()));
      wrapper->cuda_stream_ = static_cast<const CUDAContext*>(context)->cuda_stream();
      wrapper->status_ = EventStatus::kEventScheduled;
    } else {
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::kEventFailed;
    }
  }
  wrapper->cv_recorded_.notify_all();
}

void EventFinishCUDA(const Event* event) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
    while (wrapper->status_ == EventStatus::kEventInitialized) {
      wrapper->cv_recorded_.wait(lock);
    }
  }

  if (wrapper->status_ == EventStatus::kEventScheduled) {
    CUDADeviceGuard g(wrapper->cuda_gpu_id_);
    auto cudaResult = cudaEventSynchronize(wrapper->cuda_event_);
    if (cudaResult == cudaSuccess) {
      wrapper->status_ = EventStatus::kEventSuccess;
    } else {
      const auto& err_msg = cudaGetErrorString(cudaResult);

      std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::kEventFailed;
    }
  }
}

// Both waiter and event are CUDA. Non-blocking
void EventWaitCUDACUDA(const Event* event, void* context) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
    while (wrapper->status_ == EventStatus::kEventInitialized) {
      wrapper->cv_recorded_.wait(lock);
    }
  }

  if (wrapper->status_ == EventStatus::kEventScheduled) {
    auto context_stream = static_cast<CUDAContext*>(context)->cuda_stream();
    auto event_stream = wrapper->cuda_stream_;
    if (context_stream != event_stream) {
      CUDA_CHECK(cudaStreamWaitEvent(context_stream, wrapper->cuda_event_, 0));
    }
  }
}

// Waiter is CPU, event is CUDA
void EventWaitCPUCUDA(const Event* event, void* context) {
  EventFinishCUDA(event);
}

// Waiter is CUDA, event is CPU
void EventWaitCUDACPU(const Event* event, void* context) {
  event->Finish();
}

EventStatus EventQueryCUDA(const Event* event) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  if (wrapper->status_ == EventStatus::kEventScheduled) {
    auto cudaResult = cudaEventQuery(wrapper->cuda_event_);
    if (cudaResult == cudaSuccess) {
      wrapper->status_ = EventStatus::kEventSuccess;
    } else if (cudaResult != cudaErrorNotReady) {
      const auto& err_msg = cudaGetErrorString(cudaResult);

      std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::kEventFailed;
    }
  }
  return static_cast<EventStatus>(wrapper->status_.load());
}

const std::string& EventErrorMessageCUDA(const Event* event) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  if (wrapper->status_ == EventStatus::kEventFailed) {
    return wrapper->err_msg_;
  } else {
    return kNoError;
  }
}

void EventSetFinishedCUDA(const Event* event, const char* err_msg) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);

    CHECK(wrapper->status_ == EventStatus::kEventInitialized,
          "Calling SetFinished on recorded CUDA event");

    if (!err_msg) {
      wrapper->status_ = EventStatus::kEventSuccess;
    } else {
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::kEventFailed;
    }
  }
  wrapper->cv_recorded_.notify_all();
}

void EventResetCUDA(Event* event) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
  wrapper->status_ = EventStatus::kEventInitialized;
  wrapper->err_msg_ = "";
  wrapper->cuda_stream_ = nullptr;
}

// Register event related function
REGISTER_EVENT_CREATE_FUNCTION(kCUDA, EventCreateCUDA);
REGISTER_EVENT_RECORD_FUNCTION(kCUDA, EventRecordCUDA);
REGISTER_EVENT_WAIT_FUNCTION(kCUDA, kCUDA, EventWaitCUDACUDA);
REGISTER_EVENT_WAIT_FUNCTION(kCPU, kCUDA, EventWaitCPUCUDA);
REGISTER_EVENT_WAIT_FUNCTION(kCUDA, kCPU, EventWaitCUDACPU);
REGISTER_EVENT_FINISH_FUNCTION(kCUDA, EventFinishCUDA);

REGISTER_EVENT_QUERY_FUNCTION(kCUDA, EventQueryCUDA);
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(kCUDA, EventErrorMessageCUDA);
REGISTER_EVENT_SET_FINISHED_FUNCTION(kCUDA, EventSetFinishedCUDA);
REGISTER_EVENT_RESET_FUNCTION(kCUDA, EventResetCUDA);

} // namespace blaze

#endif
