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

#ifndef TDM_SERVING_UTIL_CONCURRENCY_SPIN_H_
#define TDM_SERVING_UTIL_CONCURRENCY_SPIN_H_

#include <assert.h>
#include <pthread.h>

#include "util/concurrency/scoped_locker.h"
#include "util/concurrency/check_error.h"

namespace tdm_serving {
namespace util {

class CondVar;

class SpinBase {
 public:
  void Lock() {
    CHECK_PTHREAD_ERROR(pthread_spin_lock(&spin_));
    assert(IsLocked());
  }

  bool TryLock() {
    return CHECK_PTHREAD_TRYLOCK_ERROR(pthread_spin_trylock(&spin_));
  }

  bool IsLocked() const {
    return spin_ == 0;
  }

  void Unlock() {
    assert(IsLocked());
    CHECK_PTHREAD_ERROR(pthread_spin_unlock(&spin_));
  }

 protected:
  explicit SpinBase(int32_t type) {
    CHECK_PTHREAD_ERROR(pthread_spin_init(&spin_, type));
  }

  ~SpinBase() {
    CHECK_PTHREAD_ERROR(pthread_spin_destroy(&spin_));
  }

 private:
  pthread_spinlock_t spin_;
  friend class CondVar;
};

class SharedSpin: public SpinBase {
 public:
  typedef ScopedLocker<SharedSpin> Locker;
  SharedSpin() : SpinBase(PTHREAD_PROCESS_SHARED) { }
};

class PrivateSpin: public SpinBase {
 public:
  typedef ScopedLocker<PrivateSpin> Locker;
  PrivateSpin() : SpinBase(PTHREAD_PROCESS_PRIVATE) { }
};

typedef ScopedLocker<SpinBase> SpinLocker;

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_CONCURRENCY_SPIN_H_
