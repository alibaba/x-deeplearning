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

#ifndef TDM_COMMON_H_
#define TDM_COMMON_H_

#include <string>
#include <vector>
#include <unordered_map>

namespace tdm {

std::vector<std::string>
Split(const std::string& src, const std::string& pattern);

std::unordered_map<std::string, std::string>
ParseConfig(const std::string& config);


}  // namespace tdm

#endif  // TDM_COMMON_H_
