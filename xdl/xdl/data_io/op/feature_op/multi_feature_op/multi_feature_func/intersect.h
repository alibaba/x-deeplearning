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

#include <stdint.h>

namespace xdl {
namespace io {

class Intersect {
 public:
  static int64_t CombineKey(int64_t key0, int64_t key1) {
    return key0;
  }
  static float CombineValue(float value0, float value1) {
    return value0 * value1;
  }
};

}  // namespace io
}  // namespace xdl