/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef XDL_CORE_LIB_THREAD_LOCAL_H_
#define XDL_CORE_LIB_THREAD_LOCAL_H_

#include <mutex>
#include <memory>
#include <vector>

namespace xdl {

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
  /// get a thread local singlton with bool arg
  static T* Get(bool flag) {
    static __thread T* ptr = nullptr;
    if (ptr == nullptr) {
      ptr = new T(flag);
      Singleton()->RegisterDelete(ptr);
    }
    return ptr;
  }
  /// get a thread local instance
  template<typename... Args>
  static T* Get(bool* new_created, Args &&... args) {
    static __thread T* ptr = nullptr;
    if (ptr == nullptr) {
      *new_created = true;
      ptr = new T(std::forward<Args>(args)...);
      Singleton()->RegisterDelete(ptr);
    } else {
      *new_created = false;
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

}  // namespace xdl

#endif  // XDL_CORE_LIB_THREAD_LOCAL_H_
