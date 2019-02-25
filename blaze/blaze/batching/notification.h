// Notification for batching module
//
#pragma once

#include <assert.h>
#include <chrono>
#include <condition_variable>

#include "blaze/batching/mutex.h"

namespace blaze {
namespace batching {

/// Notification for concurrency
class Notification {
 public:
  Notification() : notified_(false) {}
  ~Notification() {}

  /// Notify
  void Notify() {
    mutex_lock l(mu_);
    assert(!notified_);
    notified_ = true;
    cv_.notify_all();
  }

  /// Has been notified
  bool HasBeenNotified() {
    mutex_lock l(mu_);
    return notified_;
  }

  /// Wait for notification
  void WaitForNotification() {
    mutex_lock l(mu_);
    while (!notified_) {
      cv_.wait(l);
    }
  }

  /// Wait for notification with timeout
  bool WaitForNotificationWithTimeout(uint64_t timeout_in_ms) {
    mutex_lock l(mu_);
    std::cv_status s =
        cv_.wait_for(l, std::chrono::milliseconds(timeout_in_ms));
    return (s == std::cv_status::timeout) ? true : false;
  }

 private:
  mutex mu_;
  condition_variable cv_;
  bool notified_;
};

}  // namespace batching
}  // namespace blaze
