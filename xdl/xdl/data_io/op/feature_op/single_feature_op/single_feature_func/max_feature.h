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

namespace xdl {
namespace io {

class MaxFeature {
 public:
  static int StatisValue(float value0, float value1, float &value) {
    if (value0 < value1) {
      value = value1;
      return 1;
    } else {
      value = value0;
      return 0;
    }
  }
};

}  // namespace io
}  // namespace xdl