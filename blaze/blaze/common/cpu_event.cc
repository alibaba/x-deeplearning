/*!
 * \file cpu_event.cc
 * \brief The cpu event.
 */
#include "blaze/common/cpu_event.h"

namespace {
const std::string kNoError = "No error";
}  // namespace

namespace blaze {

void EventCreateCPU(const DeviceOption& option, Event* event) {
  event->event_ = std::make_shared<CPUEventWrapper>(option);
}

void EventRecordCPU(Event* event, const void*, const char* err_msg) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);

  CHECK(wrapper->status_ != EventStatus::kEventScheduled,
        "Calling Record multiple times");

  if (wrapper->status_ == EventStatus::kEventInitialized) {
    if (!err_msg) {
      wrapper->status_ = EventStatus::kEventScheduled;
    } else {
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::kEventFailed;
      wrapper->cv_completed_.notify_all();
    }
  }
}

void EventFinishCPU(const Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  while (wrapper->status_ != EventStatus::kEventSuccess &&
         wrapper->status_ != EventStatus::kEventFailed) {
    wrapper->cv_completed_.wait(lock);
  }
}

void EventWaitCPUCPU(const Event* event, void*) {
  EventFinishCPU(event);
}

EventStatus EventQueryCPU(const Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  return static_cast<EventStatus>(wrapper->status_.load());
}

const std::string& EventErrorMessageCPU(const Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  if (wrapper->status_ == EventStatus::kEventFailed) {
    return wrapper->err_msg_;
  } else {
    return kNoError;
  }
}

void EventSetFinishedCPU(const Event* event, const char* err_msg) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);

  CHECK(wrapper->status_ == EventStatus::kEventInitialized ||
        wrapper->status_ == EventStatus::kEventScheduled,
        "Calling SetFinished on finished event");

  if (!err_msg) {
    wrapper->status_ = EventStatus::kEventSuccess;
  } else {
    wrapper->err_msg_ = err_msg;
    wrapper->status_ = EventStatus::kEventFailed;
  }
  wrapper->cv_completed_.notify_all();
}

void EventResetCPU(Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  wrapper->status_ = EventStatus::kEventInitialized;
  wrapper->err_msg_ = "";
}

// Register
REGISTER_EVENT_CREATE_FUNCTION(kCPU, EventCreateCPU);
REGISTER_EVENT_RECORD_FUNCTION(kCPU, EventRecordCPU);
REGISTER_EVENT_WAIT_FUNCTION(kCPU, kCPU, EventWaitCPUCPU);
REGISTER_EVENT_FINISH_FUNCTION(kCPU, EventFinishCPU);

REGISTER_EVENT_QUERY_FUNCTION(kCPU, EventQueryCPU);
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(kCPU, EventErrorMessageCPU);
REGISTER_EVENT_SET_FINISHED_FUNCTION(kCPU, EventSetFinishedCPU);
REGISTER_EVENT_RESET_FUNCTION(kCPU, EventResetCPU);

}  // namespace blaze
