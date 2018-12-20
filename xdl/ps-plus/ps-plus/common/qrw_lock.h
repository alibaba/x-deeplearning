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

#ifndef PS_COMMON_QRW_LOCK_H_
#define PS_COMMON_QRW_LOCK_H_

#include <atomic>
#include <pthread.h>
#include <iostream>

namespace ps {

class QRWLock {
 public:
  QRWLock() {
    pthread_rwlockattr_t attr;
    pthread_rwlockattr_init(&attr);
    pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
    pthread_rwlock_init(&lock_, &attr);
    simple_lock_.store(0);
  }
  ~QRWLock() {
    pthread_rwlock_destroy(&lock_);
  }
  bool SimpleReadLock() {
    if ((simple_lock_++ & 0xFFFF0000) == 0) {
      return true;
    } else {
      simple_lock_--;
      return false;
    }
  }
  void ReadLock() {
    pthread_rwlock_rdlock(&lock_);
  }

  void WriteLock() {
    simple_lock_ += 0x10000;
    pthread_rwlock_wrlock(&lock_);
    while ((simple_lock_.load(std::memory_order_relaxed) & 0xFFFF) != 0);
  }

  void SimpleReadUnlock() {
    simple_lock_--;
  }

  void ReadUnlock() {
    pthread_rwlock_unlock(&lock_);
  }

  void WriteUnlock() {
    pthread_rwlock_unlock(&lock_);
    simple_lock_ -= 0x10000;
  }
 private:
  QRWLock(const QRWLock&) = delete;
  pthread_rwlock_t lock_;
  std::atomic<uint32_t> simple_lock_;
};

class QRWLocker {
 public:
  enum LockType {
    kSimpleRead,
    kRead,
    kWrite
  };
  QRWLocker(QRWLock& lock, LockType type)
    : lock_(&lock), type_(type) {
    Lock();
  }
  ~QRWLocker() {
    Unlock();
  }
  void ChangeType(LockType type) {
    Unlock();
    type_ = type;
    Lock();
  }
 private:
  QRWLocker(const QRWLocker&) = delete;
  void Lock() {
    switch (type_) {
    case kSimpleRead:
      if (!lock_->SimpleReadLock()) {
        lock_->ReadLock();
        type_ = kRead;
      }
      break;
    case kRead:
      lock_->ReadLock();
      break;
    case kWrite:
      lock_->WriteLock();
      break;
    };
  }
  void Unlock() {
    switch (type_) {
    case kSimpleRead:
      lock_->SimpleReadUnlock();
      break;
    case kRead:
      lock_->ReadUnlock();
      break;
    case kWrite:
      lock_->WriteUnlock();
      break;
    };
  }
  QRWLock* lock_;
  LockType type_;
};

}

#endif

