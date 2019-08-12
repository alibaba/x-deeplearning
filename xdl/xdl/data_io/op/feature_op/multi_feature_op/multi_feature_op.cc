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


#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_op.h"

#include <xdl/core/utils/logging.h>
#include <unordered_map>

#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature/cross_feature.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature/merge_feature.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature/vector_feature.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature/multi_feature.h"
#include "xdl/proto/sample.pb.h"

namespace xdl {
namespace io {

bool MultiFeatureOp::Init(MultiFeaOpType multi_fea_op_type,
                          CombineKeyFunc combine_key_func,
                          CombineValueFunc combine_value_func) {
  XDL_CHECK(combine_value_func != nullptr);
  multi_fea_op_type_ = multi_fea_op_type;
  switch (multi_fea_op_type) {
   case MultiFeaOpType::kCrossFeatureOp:
    XDL_CHECK(combine_key_func != nullptr);
    multi_feature_ = new CrossFeature();
    break;
   case MultiFeaOpType::kMergeFeatureOp:
    XDL_CHECK(combine_key_func != nullptr);
    multi_feature_ = new MergeFeature();
    break;
   case MultiFeaOpType::kVectorFeatureOp:
    multi_feature_ = new VectorFeature();
    break;
  }
  XDL_CHECK(multi_feature_ != nullptr);
  multi_feature_->Init(combine_key_func, combine_value_func);
  return true;
}

bool MultiFeatureOp::Destroy() {
  if (multi_feature_ == nullptr)  return true;
  switch (multi_fea_op_type_) {
   case MultiFeaOpType::kCrossFeatureOp:
    delete dynamic_cast<CrossFeature *>(multi_feature_);
    break;
   case MultiFeaOpType::kMergeFeatureOp:
    delete dynamic_cast<MergeFeature *>(multi_feature_);
    break;
   case MultiFeaOpType::kVectorFeatureOp:
    delete dynamic_cast<VectorFeature *>(multi_feature_);
    break;
  }
  multi_feature_ = nullptr;
  return true;
}

bool MultiFeatureOp::Run(std::vector<const ExprNode *> &source_nodes, ExprNode *result_node) {
  XDL_CHECK(result_node != nullptr);
  return multi_feature_->Combine(source_nodes, result_node);
}

}  // namespace io
}  // namespace xdl
