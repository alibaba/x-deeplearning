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

#ifndef TDM_SERVING_UTIL_CONCURRENCY_MUTEX_H_
#define TDM_SERVING_UTIL_CONCURRENCY_MUTEX_H_

#include <assert.h>
#include <errno.h>
#include <pthread.h>

#include "util/concurrency/scoped_locker.h"
#include "util/concurrency/check_error.h"

namespace tdm_serving {
namespace util {

class CondVar;

class MutexBase {
 public:
  void Lock() {
    CHECK_PTHREAD_ERROR(pthread_mutex_lock(&mutex_));
    assert(IsLocked());
  }

  bool TryLock() {
    return CHECK_PTHREAD_TRYLOCK_ERROR(pthread_mutex_trylock(&mutex_));
  }

  bool IsLocked() const {
    return mutex_.__data.__lock > 0;
  }

  void Unlock() {
    assert(IsLocked());
    CHECK_PTHREAD_ERROR(pthread_mutex_unlock(&mutex_));
  }

 protected:
  explicit MutexBase(int type) {
    pthread_mutexattr_t attr;
    CHECK_PTHREAD_ERROR(pthread_mutexattr_init(&attr));
    CHECK_PTHREAD_ERROR(pthread_mutexattr_settype(&attr, type));
    CHECK_PTHREAD_ERROR(pthread_mutex_init(&mutex_, &attr));
    CHECK_PTHREAD_ERROR(pthread_mutexattr_destroy(&attr));
  }

  ~MutexBase() {
    CHECK_PTHREAD_ERROR(pthread_mutex_destroy(&mutex_));
  }

 private:
  pthread_mutex_t mutex_;
  friend class CondVar;
};

// SimpleMutex is fast but non-recursive. If the same thread try to
// acquire the lock twice, deadlock would take place.
class SimpleMutex: public MutexBase {
 public:
  typedef ScopedLocker<SimpleMutex> Locker;
  SimpleMutex() : MutexBase(PTHREAD_MUTEX_DEFAULT) { }
};

// RecursiveMutex can be acquired by same thread multiple times,
// but slower than SimpleMutex
class RecursiveMutex: public MutexBase {
 public:
  typedef ScopedLocker<RecursiveMutex> Locker;
  RecursiveMutex() : MutexBase(PTHREAD_MUTEX_RECURSIVE_NP) { }
};

// AdaptiveMutex tries to spin some time and wait,
// if the lock can'tbe  acquired.
class AdaptiveMutex: public MutexBase {
 public:
  typedef ScopedLocker<AdaptiveMutex> Locker;
  AdaptiveMutex() : MutexBase(PTHREAD_MUTEX_ADAPTIVE_NP) { }
};

class Mutex: public MutexBase {
 public:
  typedef ScopedLocker<MutexBase> Locker;
  Mutex() : MutexBase(PTHREAD_MUTEX_ERRORCHECK_NP) { }
};

typedef ScopedLocker<MutexBase> MutexLocker;

// NullMutex is template mutex param placeholder.
// Don't make this class uncopyable
class NullMutex {
 public:
  typedef ScopedLocker<NullMutex> Locker;
 public:
  NullMutex() : locked_(false) { }
  void Lock() {
    locked_ = true;
  }
  bool TryLock() {
    locked_ = true;
    return true;
  }
  bool IsLocked() const {
    return locked_;
  }
  void Unlock() {
    locked_ = false;
  }

 private:
  bool locked_;
};

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_CONCURRENCY_MUTEX_H_

