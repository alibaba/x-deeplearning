// Mutex for batching module
//
#pragma once

#include <condition_variable>
#include <mutex>

namespace blaze {
namespace batching {

// A class that wraps around the std::mutex implementation
class mutex : public std::mutex {
 public:
  mutex() {}
  void lock() { std::mutex::lock(); }
  bool try_lock() {
    return std::mutex::try_lock();
  };
  void unlock() { std::mutex::unlock(); }
};

class mutex_lock : public std::unique_lock<std::mutex> {
 public:
  mutex_lock(class mutex& m) : std::unique_lock<std::mutex>(m) {}
  mutex_lock(class mutex& m, std::try_to_lock_t t)
      : std::unique_lock<std::mutex>(m, t) {}
  mutex_lock(mutex_lock&& ml) noexcept
      : std::unique_lock<std::mutex>(std::move(ml)) {}
  ~mutex_lock() {}
};

// Catch bug where variable name is omitted, e.g. mutex_lock (mu);
#define mutex_lock(x) static_assert(0, "mutex_lock_decl_missing_var_name");

using std::condition_variable;

/// 0 timeout; 1 notified
inline int WaitForMilliseconds(mutex_lock* mu,
                               condition_variable* cv, uint64_t ms) {
  std::cv_status s = cv->wait_for(*mu, std::chrono::milliseconds(ms));
  return (s == std::cv_status::timeout) ? 0 : 1;
}

}  // namespace batching
}  // namespace blaze
