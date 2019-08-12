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

#ifndef PS_SERVICE_SEASTAR_LIB_CPU_POOL_H_
#define PS_SERVICE_SEASTAR_LIB_CPU_POOL_H_

#include <mutex>
#include <sstream>

#include "ps-plus/common/net_utils.h"

namespace ps {
namespace service {
namespace seastar {

class CPUPool {
 public:
  static CPUPool* GetInstance() {
    static CPUPool instance;
    return &instance;
  }

  bool Allocate(int num, std::vector<int>* core_ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (num > total_ - use_) {
      return false;
    } else {
      core_ids->reserve(num);
      for (int i = 0; i < num; ++i) {
        core_ids->push_back(use_ + i);
      }

      use_ += num;
    }

    return true;
  }

  bool Allocate(int num, std::string* core_ids) {
    std::vector<int> ids;
    if (!Allocate(num, &ids)) {
      return false;
    }

    std::stringstream ss;
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i > 0) {
        ss << ",";
      }

      ss << ids[i];
    }

    *core_ids = ss.str();
    std::cout << "cpuset:" << *core_ids << std::endl;
    return true;
  }

 private:
  CPUPool()
    : total_(NetUtils::GetAvailableCpuNum()) 
    , use_(0) {}

 private:
  std::mutex mutex_;
  int total_;
  int use_;
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif //PS_SERVICE_SEASTAR_LIB_CPU_POOL_H_
