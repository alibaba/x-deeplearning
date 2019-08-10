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

#include "xdl/data_io/op/feature_op/feature_op_type.h"

namespace xdl {
namespace io {

enum SingleFeaOpType {
  kDefaultSingleFeatureOp = 0,
  kTraversalFeatureOp     = 1,
  kStatisFeatureOp        = 2,
};

class SingleFeatureOp {
 public:
  bool Init(SingleFeaOpType single_fea_op_type,
            TransformKeyFunc transform_key_func,
            TransformValueFunc transform_value_func,
            StatisValueFunc statis_value_func,
            bool is_average = false);
  bool Destroy();

  bool Run(const ExprNode *source_node, ExprNode *result_node);

  SingleFeaOpType type() const { return single_fea_op_type_; }
  const SingleFeature *single_feature() const { return single_feature_; }

 private:
  SingleFeaOpType single_fea_op_type_;
  SingleFeature *single_feature_ = nullptr;
};


}  // namespace io
}  // namespace xdl