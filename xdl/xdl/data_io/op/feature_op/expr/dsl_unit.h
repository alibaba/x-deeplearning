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


#pragma once

#include <string>

namespace xdl {
namespace io {

enum DslType {
  kDslDefault = 0,
  kDslNumeric = 1,
  kDslKv      = 2,
  kDslId      = 3,
};

struct DslUnit {
  std::string name;
  std::string expr;
  DslType type;
  bool IsEmpty() const { return name.empty() || expr.empty() || type == DslType::kDslDefault; }
};

}  // namespace io
}  // namespace xdl