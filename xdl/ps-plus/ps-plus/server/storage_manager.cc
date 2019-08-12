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

#include "ps-plus/server/storage_manager.h"

namespace ps {
namespace server {

Status StorageManager::Get(const std::string& name, Variable** variable) {
  QRWLocker lock(rd_lock_, QRWLocker::kSimpleRead);
  auto iter = variable_map_.find(name);
  if (iter == variable_map_.end()) {
    return Status::NotFound("Storage Manager Get: Not Found For Name: " + name);
  }
  if (iter->second == nullptr) {
    return Status::NotFound("Storage Manager Get: Initializing: " + name);
  }
  *variable = iter->second.get();
  return Status::Ok();
}

Status StorageManager::Set(const std::string& name, const std::function<Variable*()>& variable_constructor) {
  {
    QRWLocker lock(rd_lock_, QRWLocker::kWrite);
    auto iter = variable_map_.find(name);
    if (iter != variable_map_.end()) {
      return Status::AlreadyExist("Initialized or Initializing: " + name);
    }
    variable_map_[name].reset(nullptr);
  }
  Variable* variable = variable_constructor();
  if (variable == nullptr) {
    QRWLocker lock(rd_lock_, QRWLocker::kWrite);
    variable_map_.erase(name);
    return Status::ArgumentError("Storage Manager Set: Variable Constructor Return nullptr");
  }

  {
    QRWLocker lock(rd_lock_, QRWLocker::kWrite);
    variable_map_[name].reset(variable);
  }
  return Status::Ok();
}

Status StorageManager::Reset() {
  QRWLocker lock(rd_lock_, QRWLocker::kWrite);
  variable_map_.clear();
  return Status::Ok();
}

}
}

