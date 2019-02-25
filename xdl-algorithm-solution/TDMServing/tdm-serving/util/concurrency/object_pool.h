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

#ifndef TDM_SERVING_UTIL_CONCURRENCY_OBJECT_POOL_H_
#define TDM_SERVING_UTIL_CONCURRENCY_OBJECT_POOL_H_

#include <limits.h>

#include <algorithm>
#include <vector>

#include "util/concurrency/mutex.h"

#undef min
#undef max

namespace tdm_serving {
namespace util {

struct ObjectPoolBasicAllocator {
  template<typename T>
  static T *New(T*) {
  }

  template<typename T>
  static void Clear(T*) {
  }

  template<typename T>
  static void Delete(T *p) {
    delete p;
  }
};

struct ObjectPoolDefaultAllocator: public ObjectPoolBasicAllocator {
  template<typename T>
  static T *New(T*) {
    return new T();
  }
};

/// clear for google style
struct ObjectPoolNormalAllocator: public ObjectPoolDefaultAllocator {
  template<typename T>
  static void Clear(T *p) {
    p->Clear();
  }
};

/// clear for STL style
struct ObjectPoolStdCxxAllocator: public ObjectPoolDefaultAllocator {
  template<typename T>
  static void Clear(T *p) {
    p->clear();
  }
};

/// object_pool with thread safe
template<typename T,
    typename Allocator = ObjectPoolDefaultAllocator,
    typename LockType = SimpleMutex
>
class ObjectPool {
  typedef ObjectPool ThisType;

 public:
  typedef T ObjectType;

 public:
  explicit ObjectPool(
      size_t initial_size = 0,  // initial size of creating object
      size_t quota = INT_MAX,  // max size
      bool auto_create = true,
      Allocator allocator = Allocator() ) :
       quota_(quota), auto_create_(auto_create), allocator_(allocator) {
     Reserve(std::min(initial_size, quota));
  }

  ~ObjectPool() {
    Clear();
  }

  size_t GetQuota() const {
    return quota_;
  }

  size_t Size() const {
    MutexLocker locker(&lock_);
    return pooled_objects_.size();
  }

  void SetQuota(size_t size) {
    MutexLocker locker(&lock_);
    quota_ = size;
    UnlockedShrink(size);
  }

  void Reserve(size_t size) {
    if (size > quota_)
      size = quota_;
    MutexLocker locker(&lock_);
    while (pooled_objects_.size() < size) {
      pooled_objects_.push_back(NewObject());
    }
  }

  /// get one object
  T *Acquire() {
    {
      MutexLocker locker(&lock_);
      if (pooled_objects_.empty()) {
        if (!auto_create_)
          return NULL;
      } else {
        T *p = pooled_objects_.back();
        pooled_objects_.pop_back();
        return p;
      }
    }
    return NewObject();
  }

  /// release one object
  void Release(const T *p, bool is_erase = false) {
    if (p == NULL) {
      return;
    }
    T *q = const_cast<T *>(p);
    allocator_.Clear(q);
    if (!is_erase) {
      MutexLocker locker(&lock_);
      if (pooled_objects_.size() < quota_) {
        pooled_objects_.push_back(q);
        return;
      }
    }
    allocator_.Delete(q);
  }

  void Shrink(size_t size = 0) {
    MutexLocker locker(&lock_);
    UnlockedShrink(size);
  }

  void Clear() {
    Shrink(0);
  }

 private:
  void UnlockedShrink(size_t size) {
    while (pooled_objects_.size() > size) {
      allocator_.Delete(pooled_objects_.back());
      pooled_objects_.pop_back();
    }
  }

  T *NewObject() {
    return allocator_.New(static_cast<T *>(0));
  }

 private:
  ObjectPool(const ObjectPool &src);
  ObjectPool &operator=(const ObjectPool &rhs);

 private:
  mutable LockType lock_;
  std::vector<T *> pooled_objects_;
  size_t quota_;
  bool auto_create_;
  Allocator allocator_;
};

template<
    typename T,
    size_t Quota,
    typename Allocator = ObjectPoolDefaultAllocator,
    typename LockType = SimpleMutex
>
class FixedObjectPool {
  typedef FixedObjectPool ThisType;

 public:
  typedef T ObjectType;

 public:
  explicit FixedObjectPool(
      size_t initial_size = 0,
      bool auto_create = true,
      Allocator allocator = Allocator() ) :
      count_(0), auto_create_(auto_create), allocator_(allocator) {
    Reserve(std::min(Quota, initial_size));
  }

  ~FixedObjectPool() {
    Clear();
  }

  // get one object
  T *Acquire() {
    {
      MutexLocker locker(&lock_);
      if (count_ > 0) {
        return pooled_objects_[--count_];
      }
    }
    if (auto_create_) {
      return NewObject();
    } else {
      return NULL;
    }
  }

  // release one object
  void Release(const T *p) {
    if (p == NULL) {
      return;
    }
    T *q = const_cast<T *>(p);
    allocator_.Clear(q);
    {
      MutexLocker locker(&lock_);
      if (count_ < Quota) {
        pooled_objects_[count_++] = q;
        return;
      }
    }
    allocator_.Delete(q);
  }

  size_t Size() const {
    MutexLocker locker(&lock_);
    return count_;
  }

  void Reserve(size_t size) {
    if (size > Quota)
      size = Quota;
    MutexLocker locker(&lock_);
    while (count_ < size) {
      T *p = NewObject();
      pooled_objects_[count_++] = p;
    }
  }

  void Shrink(size_t size = 0) {
    MutexLocker locker(&lock_);
    UnlockedShrink(size);
  }

  void Clear() {
    MutexLocker locker(&lock_);
    UnlockedShrink(0);
  }

 private:
  FixedObjectPool(const FixedObjectPool &src);
  FixedObjectPool &operator=(const FixedObjectPool &rhs);
  T *NewObject() {
    return allocator_.New(static_cast<T *>(0));
  }

  void UnlockedShrink(size_t size) {
    while (count_ > size) {
      allocator_.Delete(pooled_objects_[--count_]);
    }
  }

 private:
  mutable LockType lock_;
  T *pooled_objects_[Quota];
  size_t count_;
  bool auto_create_;
  Allocator allocator_;
};

// DEPRECATED_BY(ObjectPoolDefaultAllocator)
typedef ObjectPoolDefaultAllocator NullClear;

// DEPRECATED_BY(ObjectPoolStdCxxAllocator)
typedef ObjectPoolStdCxxAllocator CallMember_clear;

// DEPRECATED_BY(ObjectPoolNormalAllocator)
typedef ObjectPoolNormalAllocator CallMember_Clear;

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_CONCURRENCY_OBJECT_POOL_H_
