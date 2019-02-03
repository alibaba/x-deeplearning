/*!
 * \file cpu_event.h
 * \brief The cpu event.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>

#include "blaze/common/event.h"
#include "blaze/common/exception.h"

namespace blaze {

struct CPUEventWrapper {
  explicit CPUEventWrapper(const DeviceOption& option) :
      status_(EventStatus::kEventInitialized) {
    CHECK(option.device_type() == kCPU, "Expected kCPU device type");
  }
  ~CPUEventWrapper() { }

  std::mutex mutex_;
  std::condition_variable cv_completed_;
  std::atomic<int> status_;
  std::string err_msg_;
};

void EventCreateCPU(const DeviceOption& option, Event* event);
void EventRecordCPU(Event* event, const void*, const char* err_msg);
void EventFinishCPU(const Event* event);
void EventWaitCPUCPU(const Event* event, void*);
EventStatus EventQueryCPU(const Event* event);
const std::string& EventErrorMessageCPU(const Event* event);
void EventSetFinishedCPU(const Event* event, const char* err_msg);
bool EventCanScheduleCPU(const Event*, const Event*);
void EventResetCPU(Event*);

}  // namespace blaze

