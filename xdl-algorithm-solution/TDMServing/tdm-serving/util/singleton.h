/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TDM_SERVING_UTIL_SINGLETON_H_
#define TDM_SERVING_UTIL_SINGLETON_H_

#include <stdio.h>
#include <stdlib.h>

#include "common/common_def.h"
#include "util/concurrency/mutex.h"
#include "util/concurrency/barrier.h"

namespace tdm_serving {
namespace util {

template<typename T>
class SingletonBase {
 public:
  static T& Instance() {
    if (!is_alive_) {
      fprintf(stderr, "Singleton has been destroyed\n");
      abort();
    }

    if (!instance_) {
      ScopedLocker<Mutex> locker(&lock_);
      if (!instance_) {
        T* p = new T();
        MemoryWriteBarrier();
        instance_ = p;
        atexit(Destroy);
      }
    }
    return *instance_;
  }

  template<typename A1>
  static T& Instance(const A1& a1) {
    if (!is_alive_) {
      fprintf(stderr, "Singleton has been destroyed\n");
      abort();
    }

    // Double check locking optimize
    if (!instance_) {
      ScopedLocker<Mutex> locker(&lock_);
      if (!instance_) {
        T* p = new T(a1);
        MemoryWriteBarrier();
        instance_ = p;
        atexit(Destroy);
      }
    }
    return *instance_;
  }

  static bool IsAlive() {
    return instance_ != NULL;
  }

 protected:
  SingletonBase() { }
  ~SingletonBase() { }

 private:
  static void Destroy() {
    is_alive_ = false;
    // Need not locking
    if (instance_) {
      delete instance_;
      instance_ = NULL;
    }
  }

  static Mutex lock_;
  static T* volatile instance_;
  static bool volatile is_alive_;

  DISALLOW_COPY_AND_ASSIGN(SingletonBase);
};

template<typename T>
Mutex SingletonBase<T>::lock_;

template<typename T>
T* volatile SingletonBase<T>::instance_;

template<typename T>
bool volatile SingletonBase<T>::is_alive_ = true;

template<typename T>
class Singleton: public SingletonBase<T> {
};

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_SINGLETON_H_
