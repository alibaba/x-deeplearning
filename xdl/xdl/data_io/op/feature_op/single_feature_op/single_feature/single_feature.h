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

#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_op.h"

namespace xdl {
namespace io {

class SingleFeature {
 public:
  virtual ~SingleFeature() = default;

  virtual void Init(TransformKeyFunc transform_key_func,
                    TransformValueFunc transform_value_func,
                    StatisValueFunc statis_value_func,
                    bool is_average = false) {
    transform_key_func_ = transform_key_func;
    transform_value_func_ = transform_value_func;
    statis_value_func_ = statis_value_func;
  }

  virtual bool Transform(const ExprNode *source_node, ExprNode *result_node) = 0;

  bool is_average() const { return is_average_; }

 protected:
  TransformKeyFunc transform_key_func_ = nullptr;
  TransformValueFunc transform_value_func_ = nullptr;
  StatisValueFunc statis_value_func_ = nullptr;
  bool is_average_ = false;
};

}  // namespace io
}  // namespace xdl