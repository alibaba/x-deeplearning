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

#include "xdl/data_io/op/feature_op/feature_op_constant.h"
#include "xdl/proto/sample.pb.h"

namespace xdl {
namespace io {

class FeatureUtil {
 public:
  static int64_t GetKey(const FeatureValue &feature_value) {
    return feature_value.has_key() ? feature_value.key() : kDefaultKey;
  }
  static float GetValue(const FeatureValue &feature_value) {
    return feature_value.has_value() ? feature_value.value() : kDefaultValue;
  }
};

}  // namespace io
}  // namespace xdl