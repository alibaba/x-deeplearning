// Periodic Function 
//
#include "blaze/batching/periodic_function.h"

#include "blaze/common/log.h"

namespace blaze {
namespace batching {

PeriodicFunction::PeriodicFunction(const std::function<void()>& function,
                                   const int64_t interval_micros,
                                   const Options& options)
    : function_(function),
      interval_micros_([interval_micros]() -> int64_t {
        if (interval_micros < 0) {
          LOG_ERROR("The value of interval_micros should be >= 0");
          return 0;
        }
        return interval_micros;
      }()),
      options_(options) {
  thread_.reset(Env::StartThread(options_.thread_options,
                                 options_.thread_name_prefix,
                                 [this]() {
                                   RunLoop(Env::NowMicros());
                                 }));
}

PeriodicFunction::~PeriodicFunction() {
  NotifyStop();

  /// Wait for threads to complete and clean up.
  thread_.reset();
}

void PeriodicFunction::NotifyStop() {
  if (!stop_thread_.HasBeenNotified()) {
    stop_thread_.Notify();
  }
}

void PeriodicFunction::RunLoop(const int64_t start) {
  if (options_.startup_delay_micros > 0) {
    const int64_t deadline = start + options_.startup_delay_micros;
    Env::SleepForMicroseconds(deadline - start);
  }
  while (!stop_thread_.HasBeenNotified()) {
    // DLOG(INFO) << "Running function.";
    const int64_t begin = Env::NowMicros();
    function_();

    /// Take the max() here to guard againest time going bachwards which
    /// sometimes happens in mutiproc machines.
    const int64_t end = std::max(static_cast<int64_t>(Env::NowMicros()), begin);

    /// The deadline is relative to when the last function started
    const int64_t deadline = begin + interval_micros_;

    /// We want to sleep util 'deadline'
    if (deadline > end) {
      Env::SleepForMicroseconds(deadline - end);
    } else {
      /// DLOG(INFO) << "Function took longer than interval_micros, so not
      /// sleeping";
    }
  }
}

}  // namespace batching
}  // namespace blaze
