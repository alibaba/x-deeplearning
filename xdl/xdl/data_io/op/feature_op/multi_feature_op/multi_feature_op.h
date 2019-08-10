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

enum MultiFeaOpType {
  kDefaultMultiFeatureOp = 0,
  kCrossFeatureOp        = 1,  // CartesianProduct
  kMergeFeatureOp        = 2,  // Intersect
  kVectorFeatureOp       = 3,  // DotProduct
};

class MultiFeatureOp {
 public:
  bool Init(MultiFeaOpType multi_fea_op_type,
            CombineKeyFunc combine_key_func,
            CombineValueFunc combine_value_func);
  bool Destroy();

  bool Run(std::vector<const ExprNode *> &source_nodes, ExprNode *result_node);

  MultiFeaOpType type() const { return multi_fea_op_type_; }
  const MultiFeature *multi_feature() const { return multi_feature_; }

 private:
  MultiFeaOpType multi_fea_op_type_;
  MultiFeature *multi_feature_ = nullptr;
};


}  // namespace io
}  // namespace xdl