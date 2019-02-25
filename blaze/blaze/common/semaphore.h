/*
 * \file semaphore.h 
 * \brief The semaphore 
 */
#ifndef BLAZE_COMMON_SEMAPHORE_H_
#define BLAZE_COMMON_SEMAPHORE_H_

#include <mutex>
#include <condition_variable>

namespace blaze {

class Semaphore {
 public:
  Semaphore(int count) : count_(count) {}
  Semaphore() : count_(0) {}
  
  void Init(int count) {
    count_ = count > 0 ? count : 0;
  }

  void notify() {
    std::unique_lock<std::mutex> lk(m_);
    ++count_;
    lk.unlock();
    cv_.notify_one(); 
  }    

  void wait() {
    std::unique_lock<std::mutex> lk(m_);
    while (count_ == 0) {
      // handle spurious wakeup
      cv_.wait(lk);  
    }
    --count_;
  }

  bool try_wait() {
    std::unique_lock<std::mutex> lk(m_);
    if (count_ > 0) {
      --count_;
      return true;
    } 
    return false;
  }

 private:
  int count_;
  std::mutex m_;
  std::condition_variable cv_; 
};


} // namespace blaze

#endif  // BLAZE_COMMON_SEMAPHORE_H_

