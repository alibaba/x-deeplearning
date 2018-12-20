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

#ifndef PS_SERVER_STORAGE_MANAGER_H_
#define PS_SERVER_STORAGE_MANAGER_H_

#include "ps-plus/server/variable.h"

#include "ps-plus/common/status.h"
#include "ps-plus/common/qrw_lock.h"

#include <memory>
#include <unordered_map>

namespace ps {
namespace server {

class StorageManager {
 public:
  Status Get(const std::string& name, Variable** variable);

  // We use this to avoid OOM due to multiple Init Request.
  Status Set(const std::string& name, const std::function<Variable*()>& variable_constructor);

  Status Reset();

  std::unordered_map<std::string, std::unique_ptr<Variable>>& Internal() {
    return variable_map_;
  }

 private:
  QRWLock rd_lock_;
  std::unordered_map<std::string, std::unique_ptr<Variable>> variable_map_;
};

}
}

#endif

