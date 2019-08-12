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


#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_op.h"

#include <xdl/core/utils/logging.h>

#include "xdl/data_io/op/feature_op/expr/expr_node.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature/single_feature.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature/statis_feature.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature/traversal_feature.h"
#include "xdl/proto/sample.pb.h"

namespace xdl {
namespace io {

bool SingleFeatureOp::Init(SingleFeaOpType single_fea_op_type,
                           TransformKeyFunc transform_key_func,
                           TransformValueFunc transform_value_func,
                           StatisValueFunc statis_value_func,
                           bool is_average) {
  single_fea_op_type_ = single_fea_op_type;
  switch (single_fea_op_type) {
   case SingleFeaOpType::kTraversalFeatureOp:
    XDL_CHECK(transform_value_func != nullptr);
    single_feature_ = new TraversalFeature();
    break;
   case SingleFeaOpType::kStatisFeatureOp:
    XDL_CHECK(statis_value_func != nullptr);
    single_feature_ = new StatisFeature();
    break;
  }
  XDL_CHECK(single_feature_ != nullptr);
  single_feature_->Init(transform_key_func, transform_value_func, statis_value_func, is_average);
  return true;
}

bool SingleFeatureOp::Destroy() {
  if (single_feature_ == nullptr)  return true;
  switch (single_fea_op_type_) {
   case SingleFeaOpType::kTraversalFeatureOp:
    delete dynamic_cast<TraversalFeature *>(single_feature_);
    break;
   case SingleFeaOpType::kStatisFeatureOp:
    delete dynamic_cast<StatisFeature *>(single_feature_);
    break;
  }
  single_feature_ = nullptr;
  return true;
}

bool SingleFeatureOp::Run(const ExprNode *source_node, ExprNode *result_node) {
  XDL_CHECK(source_node != nullptr);
  XDL_CHECK(result_node != nullptr);
  return single_feature_->Transform(source_node, result_node);
}

}  // namespace io
}  // namespace xdl
