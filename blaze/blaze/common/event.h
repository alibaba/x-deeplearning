/*!
 * \file event.h
 * \brief The event.
 */
#pragma once

#include <functional>
#include <memory>
#include <string>

#include "blaze/common/exception.h"
#include "blaze/proto/blaze.pb.h"

namespace blaze {

// Note: If you add a new device, let kCUDA be the last one.
constexpr int kMaxDeviceTypes = kCUDA + 1;
class Event;

enum EventStatus {
  kEventInitialized = 0,
  kEventScheduled,
  kEventSuccess,
  kEventFailed,
};

typedef void (*EventCreateFunction)(const DeviceOption&, Event*);
typedef void (*EventRecordFunction)(Event*, const void*, const char*);
typedef void (*EventWaitFunction)(const Event*, void*);
typedef void (*EventFinishFunction)(const Event*);
typedef EventStatus (*EventQueryFunction)(const Event*);
typedef const std::string& (*EventErrorMessageFunction)(const Event*);
typedef void (*EventSetFinishedFunction)(const Event*, const char*);
typedef void (*EventResetFunction)(Event*);

class Event {
 public:
  explicit Event(const DeviceOption& option) :
      event_(), type_(option.device_type()), option_(option) {
    CHECK(type_ < kMaxDeviceTypes && type_ >= 0, "type is not valid %d", type_);
    CHECK(event_creator_[type_] != nullptr, "event_creater_[%d] is nullptr", type_);
    event_creator_[type_](option, this);
  }
  ~Event() { }

  void Record(int recorder_type, const void* context, const char* err_msg = nullptr) {
    CHECK(recorder_type == type_, "You are trying to record with a wrong device type %d -> %d",
          recorder_type, type_);
    CHECK(event_recorder_[type_] != nullptr, "event_recorder_[%d] is nullptr", type_);
    event_recorder_[type_](this, context, err_msg);
  }

  void Wait(int waiter_type, void* context) const {
    CHECK(event_waiter_[waiter_type][type_] != nullptr, "event_waiter is nullptr");
    event_waiter_[waiter_type][type_](this, context);
  }

  void Finish() const {
    CHECK(event_finisher_[type_] != nullptr, "event_finisher is nullptr");
    event_finisher_[type_](this);
  }

  EventStatus Query() const {
    CHECK(event_querier_[type_], "event_querier is nullptr");
    return event_querier_[type_](this);
  }

  const std::string& ErrorMessage() const {
    CHECK(event_err_msg_getter_[type_] != nullptr, "event_err_msg_getter is nullptr");
    return event_err_msg_getter_[type_](this);
  }

  void Reset() {
    CHECK(event_resetter_[type_] != nullptr, "event_resetter is nullptr");
    event_resetter_[type_](this);
  }

  const DeviceOption& GetDeviceOption() const {
    return option_;
  }

  bool IsScheduled() const {
    auto status = Query();
    return status == EventStatus::kEventSuccess ||
        status == EventStatus::kEventFailed;
  }

  void SetFinished(const char* err_msg = nullptr) {
    CHECK(event_finished_setter_[type_] != nullptr, "event_finished_setter is nullptr");
    return event_finished_setter_[type_](this, err_msg);
  }

  // If parent op has succeeded, then we can run any child op.
  bool CanSchedule(const Event& child_event, bool supports_async) const {
    return CanSchedule(type_, Query(), child_event.GetType(), supports_async);
  }

  static bool CanSchedule(int parent_type, EventStatus parent_status,
                          int child_type, bool child_supports_async) {
    if (parent_status == EventStatus::kEventSuccess) {
      return true;
    }
    if (parent_status == EventStatus::kEventScheduled) {
      return (parent_type == child_type) && child_supports_async;
    }
    return false;
  }

  int GetType() const {
    return type_;
  }

  std::shared_ptr<void> event_;

 private:
  int type_;
  DeviceOption option_;

  static EventCreateFunction event_creator_[kMaxDeviceTypes];
  static EventRecordFunction event_recorder_[kMaxDeviceTypes];
  static EventWaitFunction event_waiter_[kMaxDeviceTypes][kMaxDeviceTypes];
  static EventFinishFunction event_finisher_[kMaxDeviceTypes];
  static EventQueryFunction event_querier_[kMaxDeviceTypes];
  static EventErrorMessageFunction event_err_msg_getter_[kMaxDeviceTypes];
  static EventSetFinishedFunction event_finished_setter_[kMaxDeviceTypes];
  static EventResetFunction event_resetter_[kMaxDeviceTypes];

  template <int d>
  friend struct EventCreateFunctionRegisterer;
  template <int d>
  friend struct EventRecordFunctionRegisterer;
  template <int w, int d>
  friend struct EventWaitFunctionRegisterer;
  template <int d>
  friend struct EventFinishFunctionRegisterer;

  template <int d>
  friend struct EventQueryFunctionRegisterer;
  template <int d>
  friend struct EventErrorMessageFunctionRegisterer;
  template <int d>
  friend struct EventSetFinishedFunctionRegisterer;
  template <int d>
  friend struct EventResetFunctionRegisterer;
};

template <int d>
struct EventCreateFunctionRegisterer {
  explicit EventCreateFunctionRegisterer(EventCreateFunction f) {
    static_assert(d < kMaxDeviceTypes, "");
    Event::event_creator_[d] = f;
  }
};
#define REGISTER_EVENT_CREATE_FUNCTION(d, f)                                     \
    namespace {                                                                  \
      static EventCreateFunctionRegisterer<d> g_event_create_##d(f);             \
    }

template <int d>
struct EventRecordFunctionRegisterer {
  explicit EventRecordFunctionRegisterer(EventRecordFunction f) {
    static_assert(d < kMaxDeviceTypes, "");
    Event::event_recorder_[d] = f;
  }
};
#define REGISTER_EVENT_RECORD_FUNCTION(d, f)                                     \
    namespace {                                                                  \
      static EventRecordFunctionRegisterer<d> g_event_record_##d(f);             \
    }

template <int waiter_type, int event_type>
struct EventWaitFunctionRegisterer {
  explicit EventWaitFunctionRegisterer(EventWaitFunction f) {
    static_assert(waiter_type < kMaxDeviceTypes, "");
    static_assert(event_type < kMaxDeviceTypes, "");
    Event::event_waiter_[waiter_type][event_type] = f;
  }
};
#define REGISTER_EVENT_WAIT_FUNCTION(w, d, f)                                    \
    namespace {                                                                  \
      static EventWaitFunctionRegisterer<w, d> g_event_wait_##w##_##d(f);        \
    }

template <int d>
struct EventQueryFunctionRegisterer {
  explicit EventQueryFunctionRegisterer(EventQueryFunction f) {
    static_assert(d < kMaxDeviceTypes, "");
    Event::event_querier_[d] = f;
  }
};
#define REGISTER_EVENT_QUERY_FUNCTION(d, f)                                       \
    namespace {                                                                   \
      static EventQueryFunctionRegisterer<d> g_event_query_##d(f);                \
    }

template <int d>
struct EventErrorMessageFunctionRegisterer {
  explicit EventErrorMessageFunctionRegisterer(EventErrorMessageFunction f) {
    static_assert(d < kMaxDeviceTypes, "");
    Event::event_err_msg_getter_[d] = f;
  }
};
#define REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(d, f)                               \
    namespace {                                                                   \
      static EventErrorMessageFunctionRegisterer<d> g_event_err_msg_##d(f);       \
    }

template <int d>
struct EventSetFinishedFunctionRegisterer {
  explicit EventSetFinishedFunctionRegisterer(EventSetFinishedFunction f) {
    static_assert(d < kMaxDeviceTypes, "");
    Event::event_finished_setter_[d] = f;
  }
};
#define REGISTER_EVENT_SET_FINISHED_FUNCTION(d, f)                                \
    namespace {                                                                   \
      static EventSetFinishedFunctionRegisterer<d> g_event_set_finished_##d(f);   \
    }

template <int d>
struct EventFinishFunctionRegisterer {
  explicit EventFinishFunctionRegisterer(EventFinishFunction f) {
    static_assert(d < kMaxDeviceTypes, "");
    Event::event_finisher_[d] = f;
  }
};
#define REGISTER_EVENT_FINISH_FUNCTION(d, f)                                      \
    namespace {                                                                   \
      static EventFinishFunctionRegisterer<d> g_event_finish_##d(f);              \
    }

template <int d>
struct EventResetFunctionRegisterer {
  explicit EventResetFunctionRegisterer(EventResetFunction f) {
    static_assert(d < kMaxDeviceTypes, "");
    Event::event_resetter_[d] = f;
  }
};
#define REGISTER_EVENT_RESET_FUNCTION(d, f)                                       \
    namespace {                                                                   \
      static EventResetFunctionRegisterer<d> g_event_reset_##d(f);                \
    }

}  // namespace blaze
