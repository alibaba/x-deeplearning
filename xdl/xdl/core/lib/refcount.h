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

#ifndef XDL_CORE_LIB_REFCOUNT_H_
#define XDL_CORE_LIB_REFCOUNT_H_

#include <atomic>
#include <utility>

namespace xdl {

class RefCounted {
 public:
  explicit RefCounted(int ref = 1) : ref_(ref) {}
  virtual ~RefCounted() {}
  void Ref() {
    ref_++;
  }
  void UnRef() {
    if (--ref_ == 0) {
      delete this;
    }
  }
 private:
  std::atomic<int64_t> ref_;
};

template<typename T>
class RefCountedPtr {
 public:
  RefCountedPtr() : ptr_(nullptr) {}

  explicit RefCountedPtr(T* ptr) : ptr_(ptr) {
    Ref();
  }

  RefCountedPtr(const RefCountedPtr& rptr) : ptr_(rptr.ptr_) {
    Ref();
  }

  RefCountedPtr(RefCountedPtr&& rptr) : ptr_(rptr.ptr_) {
    rptr.ptr_ = nullptr;
  }

  RefCountedPtr& operator=(T* ptr) {
    UnRef();
    ptr_ = ptr;
    Ref();
    return *this;
  }

  RefCountedPtr& operator=(const RefCountedPtr& rptr) {
    UnRef();
    ptr_ = rptr.ptr_;
    Ref();
    return *this;
  }

  RefCountedPtr& operator=(RefCountedPtr&& rptr) {
    std::swap(ptr_, rptr.ptr_);
    return *this;
  }

  ~RefCountedPtr() {
    if (ptr_ != nullptr) {
      ptr_->UnRef();
    }
  }

  std::add_lvalue_reference<T> operator*() const {
    return *ptr_;
  }

  T* operator->() const {
    return ptr_;
  }

  T* get() const {
    return ptr_;
  }

  template <typename... Targs>
  static RefCountedPtr Create(Targs&&... args) {
    return RefCountedPtr(new T(std::forward<Targs>(args)...), 0);
  }

 private:
  // for RefCountedPtr::Create
  RefCountedPtr(T* ptr, int x) : ptr_(ptr) {
    (void)x;
  }
  void Ref() {
    if (ptr_ != nullptr) {
      ptr_->Ref();
    }
  }

  void UnRef() {
    if (ptr_ != nullptr) {
      ptr_->UnRef();
    }
  }

  T* ptr_;
};

}  // namespace xdl

#endif  // XDL_CORE_LIB_REFCOUNT_H_
