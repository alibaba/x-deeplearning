// Env for batching module 
//
#pragma once

#include <stdint.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sched.h>
#include <pthread.h>

#include <string>
#include <thread>

namespace blaze {
namespace batching {

class Thread {
 public:
  Thread() {}

  /// Blocks until the thread of control stops running.
  virtual ~Thread() { }

 private:
  /// No copying allowed
  Thread(const Thread&);
  void operator=(const Thread&);
};
  
/// Thread options
struct ThreadOptions {
  /// Thread stack size to use (in bytes).
  size_t stack_size = 0;  // 0: use system default value
  /// Guard area size to use near thread stacks to use (in bytes)
  size_t guard_size = 0;  // 0: use system default value
};

/// Standard Thread
class StdThread : public Thread {
 public:
  /// name and thread_options are both ignored.
  StdThread(const ThreadOptions& thread_options, const std::string& name,
            std::function<void()> fn)
      : thread_(fn) { }
  ~StdThread() { thread_.join(); }

 private:
  std::thread thread_;
};

/// Env for batching
class Env {
 public:
  /// Get current micro second
  static uint64_t NowMicros() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
  }

  /// Sleep for microseconds
  static void SleepForMicroseconds(int64_t micros) {
    while (micros > 0) {
      timespec sleep_time;
      sleep_time.tv_sec = 0;
      sleep_time.tv_nsec = 0;

      if (micros >= 1e6) {
        sleep_time.tv_sec =
            std::min<int64_t>(micros / 1e6, std::numeric_limits<time_t>::max());
        micros -= static_cast<int64_t>(sleep_time.tv_sec) * 1e6;
      }
      if (micros < 1e6) {
        sleep_time.tv_nsec = 1000 * micros;
        micros = 0;
      }
      while (nanosleep(&sleep_time, &sleep_time) != 0 && errno == EINTR) {
        // Ignore signals and wait for the full interval to elapse.
      }
    }
  }
  
  /// Start standard thread
  static Thread* StartThread(const ThreadOptions& thread_options, const std::string& name,
                             std::function<void()> fn) {
    return new StdThread(thread_options, name, fn);
  }

  /// Returns num of cpus
  static int NumSchedulableCPUs() {
    cpu_set_t cpuset;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == 0) {
      return CPU_COUNT(&cpuset);
    }
    perror("sched_getaffinity");
  }
};

}  // namespace batching
}  // namespace blaze
