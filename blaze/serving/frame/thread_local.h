/*
 * \file thread_local.h
 * \brief The thread local vars
*/
#pragma once

#include <mutex>
#include <memory>
#include <vector>

namespace serving {

/// A threadlocal store to store threadlocal variables.
template <typename T>
class ThreadLocalStore {
 public:
  /// get a thread local singleton
  static T* Get() {
    static __thread T* ptr = nullptr;
    if (ptr == nullptr) {
      ptr = new T();
      Singleton()->RegisterDelete(ptr);
    }
    return ptr;
  }
  /// contructor
  ThreadLocalStore() {
    for (size_t i = 0; i < data_.size(); ++i) {
      delete data_[i];
    }
  }
  ~ThreadLocalStore() {
    for (size_t i = 0; i < data_.size(); ++i) {
      delete data_[i];
    }
  }

 private:
  /// singleton of store
  static ThreadLocalStore<T>* Singleton() {
    static ThreadLocalStore<T> inst;
    return &inst;
  }
  /// register str for internal deletion
  void RegisterDelete(T* str) {
    mutex_.lock();
    data_.push_back(str);
    mutex_.unlock();
  }

  /// internal mutex
  std::mutex mutex_;
  /// internal data
  std::vector<T*> data_;
};

}  // namespace serving
