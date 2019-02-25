// Periodic Function 
//
#pragma once

#include "blaze/batching/env.h"
#include "blaze/batching/mutex.h"
#include "blaze/batching/notification.h"

namespace blaze {
namespace batching {

class PeriodicFunction {
 public:
  /// Options for PeriodicFunction
  struct Options {
    Options() { }

    /// Any standard thread option, such as stack size, should be passed via
    /// thread_options.
    ThreadOptions thread_options;

    /// Specifies the thread name prefix
    std::string thread_name_prefix = "periodic_function";

    /// Specifies the length of sleep before the first invocation of the
    /// function.
    int64_t startup_delay_micros = 0;
  };

  /// Start the background thread which will be calling the function
  PeriodicFunction(const std::function<void()>& function,
                   int64_t interval_micros,
                   const Options& options = Options());
  ~PeriodicFunction();

 private:
  /// Notifies the background thread to stop
  void NotifyStop();

  /// (Blocking.) Loops forever calling "function_" every "interval_micros_"
  void RunLoop(int64_t start);

  /// Actual client function
  const std::function<void()> function_;
  /// Interval between calls
  const int64_t interval_micros_;

  const Options options_;
  mutable mutex mu_;
  /// Used to notify the threads to stop
  Notification stop_thread_;

  /// Thread for running "function_"
  std::unique_ptr<Thread> thread_;
};

}  // namespace batching
}  // namespace blaze
