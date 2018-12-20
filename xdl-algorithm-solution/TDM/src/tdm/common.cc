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

#include "tdm/common.h"

namespace tdm {

std::vector<std::string>
Split(const std::string& src, const std::string& pattern) {
  std::vector<std::string> result;
  size_t start = 0;
  auto index = src.find(pattern, start);
  while (index != std::string::npos) {
    result.push_back(src.substr(start, index - start));
    start = index + pattern.size();
    index = src.find(pattern, start);
  }
  result.push_back(src.substr(start));
  return result;
}

std::unordered_map<std::string, std::string>
ParseConfig(const std::string& config) {
  std::unordered_map<std::string, std::string> conf;
  auto vec = Split(config, ";");
  for (auto it = vec.begin(); it != vec.end(); ++it) {
    auto kv = Split(*it, "=");
    if (kv.size() != 2) {
      vec.clear();
      break;
    }
    conf.insert(std::make_pair(kv[0], kv[1]));
  }
  return conf;
}

}  // namespace tdm
