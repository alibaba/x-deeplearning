/*
 * Copyright 1999-2018 Alibaba Group.
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
#ifndef XDL_CORE_LIB_OBJECT_POOL_H_
#define XDL_CORE_LIB_OBJECT_POOL_H_

#include <condition_variable>
#include <mutex>
#include <vector>
#include <unordered_map>

#include "xdl/core/utils/logging.h"

namespace xdl {

template <typename T>
class ObjectPool {
 public:
  ObjectPool(size_t capacity = 0) {
    for (size_t i = 0; i < capacity; ++i) {
      T* d = new T();
      objects_.push_back(d);
    }
  }
  virtual ~ObjectPool() {
    typename std::vector<T*>::iterator iter = objects_.begin();
    for (; iter != objects_.end(); ++iter) {
      if (*iter) delete *iter;
    }
  }
  /// Acquire one object
  virtual T* Acquire() {
    std::unique_lock<std::mutex> lck(mutex_);
    if (objects_.empty()) return  new T();
    T* object = objects_.back();
    objects_.pop_back();
    return object;
  }
  /// Release one object
  virtual void Release(T* object) {
    std::unique_lock<std::mutex> lck(mutex_);
    objects_.push_back(object);
  }
  /// Release the objects vector
  virtual void Release(const std::vector<T*>& objects) {
    std::unique_lock<std::mutex> lck(mutex_);
    typename std::vector<T*>::const_iterator iter = objects.begin();
    for (; iter != objects.end(); ++iter) {
      objects_.push_back(*iter);
    }
  }
  /// Return size of current available objects
  size_t Size() {
    std::unique_lock<std::mutex> lck(mutex_);
    return objects_.size();
  }

 protected:
  std::vector<T*> objects_;
  std::mutex mutex_;
  size_t capacity_;
};

template <typename T>
class MultiObjectPool {
 public:
  virtual ~MultiObjectPool() {
    std::unique_lock<std::mutex> lck(mutex_);
    for (auto iter : kvs_) delete iter.second;
  }

  /// Acquire one object with name
  virtual T* Acquire(const std::string& name) {
    std::unique_lock<std::mutex> lck(mutex_);
    auto iter = kvs_.find(name);
    if (iter != kvs_.end()) return iter->second->Acquire();
    else return new T();
  }

  /// Release the object with name
  virtual void Release(const std::string& name, T* object) {
    std::unique_lock<std::mutex> lck(mutex_);
    auto iter = kvs_.find(name);
    if (iter != kvs_.end()) iter->second->Release(object);
    else {
      ObjectPool<T>* new_object_pool = new ObjectPool<T>();
      new_object_pool->Release(object);
      kvs_.insert(std::make_pair(name, new_object_pool));
    }
  }

 protected:
  std::unordered_map<std::string, ObjectPool<T>*> kvs_;
  std::mutex mutex_;
};

}  // namespace xdl

#endif  // XDL_CORE_LIB_OBJECT_POOL_H_
