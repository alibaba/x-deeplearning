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

#ifndef PS_COMMON_MEMGUARD_H_
#define PS_COMMON_MEMGUARD_H_

#include <cstring>

namespace ps {
namespace serializer {

class MemGuard {
 public:
  MemGuard()
    : state_(new State()) {
  }

  ~MemGuard() {
    UnRef();
  }

  MemGuard(const MemGuard& rhs) 
    : state_(rhs.state_) {
    Ref();
  }

  MemGuard(MemGuard&& rhs)
    : state_(rhs.state_) {
    rhs.state_ = nullptr;
  }

  MemGuard& operator=(const MemGuard& rhs) {
    UnRef();
    state_ = rhs.state_;
    Ref();
    return *this;
  }

  MemGuard& operator=(MemGuard&& rhs) {  
    std::swap(state_, rhs.state_);
    return *this;
  }

  template <typename T>
  T* AllocateElement(const T& data) {
    char* buf = new char[sizeof(T)];
    std::memcpy(buf, reinterpret_cast<const void*>(&data), sizeof(T));
    Collect(buf);
    return reinterpret_cast<T*>(buf);
  }

  char* AllocateBuffer(size_t len) {
    char* buf = new char[len];
    Collect(buf);
    return buf;
  }

 private:
  void Collect(char* buf) {
    state_->bufs_.push_back(buf);
  }

  void Ref() {
    if (state_ == nullptr) return;
    state_->ref_++;
  }

  void UnRef() {
    if (state_ == nullptr) return;    
    if (--state_->ref_ == 0) {
      delete state_;
      state_ = nullptr;
    }
  }
  
  struct State {
    State(): ref_(1) { 
      bufs_.reserve(32); 
    }

    ~State() {
      for (char* buf: bufs_) {
        delete[] buf;
      }
    }
    
    std::vector<char*> bufs_;
    size_t ref_;
  };

  State* state_;
};

} // serializer
} // ps

#endif // PS_COMMON_MEMGUARD_H_

