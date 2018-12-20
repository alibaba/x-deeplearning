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

// Copyright 2018 Alibaba Inc. All Rights Reserved.

#ifndef TDM_LOCAL_STORE_H_
#define TDM_LOCAL_STORE_H_

#include <string>
#include <vector>
#include <unordered_map>

#include "tdm/store.h"

namespace tdm {

class LocalStore: public Store {
 public:
  LocalStore();
  virtual ~LocalStore();

  bool Init(const std::string& config) override;

  bool Dump(const std::string& filename) override;

  bool Get(const std::string& key, std::string* value) override;
  bool Put(const std::string& key, const std::string& value) override;

  std::vector<bool>
  MGet(const std::vector<std::string>& keys,
       std::vector<std::string>* values) override;

  std::vector<bool>
  MPut(const std::vector<std::string>& keys,
       const std::vector<std::string>& values) override;

  bool Remove(const std::string& key) override;

 private:
  std::unordered_map<std::string, std::string> data_;
};

}  // namespace tdm

#endif  // TDM_LOCAL_STORE_H_
