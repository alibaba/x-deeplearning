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

#include "xdl/data_io/op/feature_op/single_feature_op/single_feature/single_feature.h"

namespace xdl {
namespace io {

class StatisFeature : public SingleFeature {
 public:
  virtual void Init(TransformKeyFunc transform_key_func,
                    TransformValueFunc transform_value_func,
                    StatisValueFunc statis_value_func,
                    bool is_average = false) override;
  virtual bool Transform(const ExprNode *source_node, ExprNode *result_node) override;

 protected:
  float Statis(const ExprNode *source_node, int64_t &key);
};

}  // namespace io
}  // namespace xdl