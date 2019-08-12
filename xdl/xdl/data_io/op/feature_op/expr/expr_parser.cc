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


#include "xdl/data_io/op/feature_op/expr/expr_parser.h"

#include <xdl/core/utils/logging.h>

#include "xdl/data_io/op/feature_op/string_util.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_op_factory.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_func/log_feature.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_func/max_feature.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_func/min_feature.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_func/sqrt_feature.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_func/sum_feature.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_op_factory.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_func/cartesian_product.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_func/dot_product.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_func/intersect.h"
#include "xdl/data_io/op/feature_op/source_feature_op/source_feature_op_factory.h"

namespace xdl {
namespace io {

void ExprParser::Parse(const std::string &expr, ExprNode &node, bool is_output) {
  std::vector<std::string> sub_exprs;
  ParseNode(expr, sub_exprs, node, is_output);  // sub_exprs, node {type, output, op, result}
  for (std::string &sub_expr : sub_exprs) {
    ExprNode pre_node;
    Parse(sub_expr, pre_node, false);        // pre_node
    node.pres.push_back(internal_nodes_->size());     // node {pres}
    internal_nodes_->push_back(std::move(pre_node));  // internal_nodes_
  }
  node.pre_nodes.resize(node.pres.size());
}

void ExprParser::ParseNode(const std::string &expr,
                           std::vector<std::string> &sub_exprs, ExprNode &node,
                           bool is_output) {
  std::vector<std::string> subs;
  StringUtil::SplitFirst(expr, subs, "(");
  if (subs.size() == 1) {
    XDL_CHECK(is_output == false);
    node.type = FeaOpType::kSourceFeatureOp;
    node.op = SourceFeatureOpFactory::Get(subs[0]);
    feature_name_vec_.push_back(std::move(subs[0]));
  } else {
    XDL_CHECK(subs.size() == 2);
    XDL_CHECK(subs[1].back() == ')');
    subs[1].erase(subs[1].size() - 1);
    if (subs[0] == "value") {
      return ParseNode(subs[1], sub_exprs, node, is_output);
    } else if (subs[0] == "cartesian") {
      node.type = FeaOpType::kMultiFeatureOp;
      node.op = MultiFeatureOpFactory::Get(MultiFeaOpType::kCrossFeatureOp,
                                           CartesianProduct::CombineKey, CartesianProduct::CombineValue);
    } else if (subs[0] == "match" || subs[0] == "intersection") {
      node.type = FeaOpType::kMultiFeatureOp;
      node.op = MultiFeatureOpFactory::Get(MultiFeaOpType::kMergeFeatureOp,
                                           Intersect::CombineKey, Intersect::CombineValue);
    } else if (subs[0] == "euclidean") {
      node.type = FeaOpType::kMultiFeatureOp;
      node.op = MultiFeatureOpFactory::Get(MultiFeaOpType::kVectorFeatureOp,
                                           nullptr, DotProduct::CombineValue);
    } else if (subs[0] == "log") {
      node.type = FeaOpType::kSingleFeatureOp;
      node.op = SingleFeatureOpFactory::Get(SingleFeaOpType::kTraversalFeatureOp,
                                            LogFeature::TransformKey, LogFeature::TransformValue,
                                            nullptr, false);
    } else if (subs[0] == "sum" || subs[0] == "count") {
      node.type = FeaOpType::kSingleFeatureOp;
      node.op = SingleFeatureOpFactory::Get(SingleFeaOpType::kStatisFeatureOp,
                                            nullptr, nullptr,
                                            SumFeature::StatisValue, false);
    } else if (subs[0] == "logsum") {
      node.type = FeaOpType::kSingleFeatureOp;
      node.op = SingleFeatureOpFactory::Get(SingleFeaOpType::kStatisFeatureOp,
                                            LogFeature::TransformKey, nullptr,
                                            SumFeature::StatisValue, false);
    } else if (subs[0] == "avg") {
      node.type = FeaOpType::kSingleFeatureOp;
      node.op = SingleFeatureOpFactory::Get(SingleFeaOpType::kStatisFeatureOp,
                                            nullptr, nullptr,
                                            SumFeature::StatisValue, true);
    } else if (subs[0] == "logavg") {
      node.type = FeaOpType::kSingleFeatureOp;
      node.op = SingleFeatureOpFactory::Get(SingleFeaOpType::kStatisFeatureOp,
                                            LogFeature::TransformKey, nullptr,
                                            SumFeature::StatisValue, true);
    } else if (subs[0] == "max") {
      node.type = FeaOpType::kSingleFeatureOp;
      node.op = SingleFeatureOpFactory::Get(SingleFeaOpType::kStatisFeatureOp,
                                            nullptr, nullptr,
                                            MaxFeature::StatisValue, false);
    } else if (subs[0] == "logmax") {
      node.type = FeaOpType::kSingleFeatureOp;
      node.op = SingleFeatureOpFactory::Get(SingleFeaOpType::kStatisFeatureOp,
                                            LogFeature::TransformKey, nullptr,
                                            MaxFeature::StatisValue, false);
    } else if (subs[0] == "min") {
      node.type = FeaOpType::kSingleFeatureOp;
      node.op = SingleFeatureOpFactory::Get(SingleFeaOpType::kStatisFeatureOp,
                                            nullptr, nullptr,
                                            MinFeature::StatisValue, false);
    } else if (subs[0] == "logmin") {
      node.type = FeaOpType::kSingleFeatureOp;
      node.op = SingleFeatureOpFactory::Get(SingleFeaOpType::kStatisFeatureOp,
                                            LogFeature::TransformKey, nullptr,
                                            MinFeature::StatisValue, false);
    } else {
      XDL_CHECK(false);
    }
    StringUtil::SplitExclude(subs[1], sub_exprs, ',', '(', ')');
    if (node.type == FeaOpType::kSingleFeatureOp) {
      XDL_CHECK(sub_exprs.size() == 1);
    } else if (node.type == FeaOpType::kMultiFeatureOp) {
      XDL_CHECK(sub_exprs.size() > 1);
    } else {
      XDL_CHECK(false);
    }
  }
  node.output = is_output;
  node.InitInternal();
  node.InitResult();
}

}  // namespace io
}  // namespace xdl
