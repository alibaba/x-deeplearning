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

#ifndef TDM_SERVING_UTIL_CONCURRENCY_SCOPED_LOCKER_H_
#define TDM_SERVING_UTIL_CONCURRENCY_SCOPED_LOCKER_H_

namespace tdm_serving {
namespace util {

template<typename LockType>
class ScopedLocker {
 public:
  explicit ScopedLocker(LockType* lock)
      : lock_(lock) {
    lock_->Lock();
  }

  ~ScopedLocker() {
    lock_->Unlock();
  }

 private:
  LockType* lock_;
};

template<typename LockType>
class ScopedTryLocker {
 public:
  explicit ScopedTryLocker(LockType* lock)
      : lock_(lock) {
    locked_ = lock_->TryLock();
  }

  ~ScopedTryLocker() {
    if (locked_)
      lock_->Unlock();
  }

  bool IsLocked() const {
    return locked_;
  }

 private:
  LockType* lock_;
  bool locked_;
};

template<typename LockType>
class ScopedReaderLocker {
 public:
  explicit ScopedReaderLocker(LockType* lock)
      : lock_(lock) {
    lock_->ReaderLock();
  }

  ~ScopedReaderLocker() {
    lock_->ReaderUnlock();
  }

 private:
  LockType* lock_;
};

template<typename LockType>
class ScopedTryReaderLocker {
 public:
  explicit ScopedTryReaderLocker(LockType* lock)
      : lock_(lock) {
    locked_ = lock_->TryReaderLock();
  }

  ~ScopedTryReaderLocker() {
    if (locked_)
      lock_->ReaderUnlock();
  }

  bool IsLocked() const {
    return locked_;
  }

 private:
  LockType* lock_;
  bool locked_;
};

template<typename LockType>
class ScopedWriterLocker {
 public:
  explicit ScopedWriterLocker(LockType* lock) : lock_(lock) {
    lock_->WriterLock();
  }

  ~ScopedWriterLocker() {
    lock_->WriterUnlock();
  }

 private:
  LockType* lock_;
};

template<typename LockType>
class ScopedTryWriterLocker {
 public:
  explicit ScopedTryWriterLocker(LockType* lock)
      : lock_(lock) {
    locked_ = lock_->TryWriterLock();
  }

  ~ScopedTryWriterLocker() {
    if (locked_)
      lock_->WriterUnlock();
  }

  bool IsLocked() const {
    return locked_;
  }

 private:
  LockType* lock_;
  bool locked_;
};

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_CONCURRENCY_SCOPED_LOCKER_H_
