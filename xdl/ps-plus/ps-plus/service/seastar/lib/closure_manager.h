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

#ifndef PS_SERVICE_SEASTAR_LIB_CLOSURE_MANAGER_H_
#define PS_SERVICE_SEASTAR_LIB_CLOSURE_MANAGER_H_

#include <functional>
#include <mutex>

#include <core/ps_common.hh>
#include "common.h"

namespace ps {
namespace service {
namespace seastar {

class Closure;

class SeastarClosureManager {
 public:
  static SeastarClosureManager* GetInstance() {
    static SeastarClosureManager sInstance;
    return &sInstance;
  }

  static uint64_t GenCallBackID(int32_t server_id, int64_t sequence_id) {
    uint64_t tmp = server_id;
    return sequence_id | (tmp << 50);
  }

  int Put(uint64_t id, Closure* closure) {
    std::lock_guard<std::mutex> lock(lock_);
    bool inserted = closures_.insert({id, closure}).second;
    return inserted ? 0 : -1;
  }

  int Get(uint64_t id, Closure** closure) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = closures_.find(id);
    if (it == closures_.end()) {
      return -1;
    }

    *closure = it->second;
    closures_.erase(it);
    return 0;
  }

 private:
  SeastarClosureManager() {}
  
 private:
  std::unordered_map<uint64_t, Closure*> closures_;
  std::mutex lock_;
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif //PS_SERVICE_SEASTAR_LIB_CLOSURE_MANAGER_H_
