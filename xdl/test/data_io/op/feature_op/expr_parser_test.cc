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

#include "gtest/gtest.h"

#include <stdio.h>
#include <string>

#include "xdl/data_io/op/feature_op/expr/expr_parser.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_op.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature/cross_feature.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature/merge_feature.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature/vector_feature.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_op.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature/statis_feature.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature/traversal_feature.h"
#include "xdl/data_io/op/feature_op/source_feature_op/source_feature_op.h"

using xdl::io::ExprNode;
using xdl::io::FeaOpType;

using xdl::io::SingleFeaOpType;
using xdl::io::MultiFeaOpType;

void CheckExprNode(const ExprNode &node, size_t pres_size,
                   FeaOpType type, bool output,
                   SingleFeaOpType single_type = SingleFeaOpType::kDefaultSingleFeatureOp,
                   bool is_average = false,
                   MultiFeaOpType multi_type = MultiFeaOpType::kDefaultMultiFeatureOp) {
  using xdl::io::SourceFeatureOp;
  using xdl::io::SingleFeatureOp;
  using xdl::io::MultiFeatureOp;

  using xdl::io::SingleFeature;
  using xdl::io::TraversalFeature;
  using xdl::io::StatisFeature;

  using xdl::io::MultiFeature;
  using xdl::io::CrossFeature;
  using xdl::io::MergeFeature;
  using xdl::io::VectorFeature;

  EXPECT_EQ(node.type, type);
  EXPECT_EQ(node.output, output);
  EXPECT_EQ(node.pres.size(), pres_size);

  EXPECT_NE(node.op, nullptr);
  if (type == FeaOpType::kSourceFeatureOp) {
    EXPECT_EQ(pres_size, 0);
    EXPECT_EQ(node.result, nullptr);
    SourceFeatureOp *op = reinterpret_cast<SourceFeatureOp *>(node.op);
  } else if (type == FeaOpType::kSingleFeatureOp) {
    EXPECT_EQ(pres_size, 1);
    EXPECT_NE(node.result, nullptr);
    SingleFeatureOp *op = reinterpret_cast<SingleFeatureOp *>(node.op);
    const SingleFeature *single_feature = op->single_feature();
    EXPECT_EQ(single_feature->is_average(), is_average);
    EXPECT_EQ(op->type(), single_type);
    if (single_type == SingleFeaOpType::kTraversalFeatureOp) {
      const TraversalFeature *traversal_feature = dynamic_cast<const TraversalFeature *>(single_feature);
      EXPECT_NE(traversal_feature, nullptr);
    } else if (single_type == SingleFeaOpType::kStatisFeatureOp) {
      const StatisFeature *statis_feature = dynamic_cast<const StatisFeature *>(single_feature);
      EXPECT_NE(statis_feature, nullptr);
    } else {
      EXPECT_TRUE(false);
    }
  } else if (node.type == FeaOpType::kMultiFeatureOp) {
    EXPECT_GT(pres_size, 1);
    EXPECT_NE(node.result, nullptr);
    MultiFeatureOp *op = reinterpret_cast<MultiFeatureOp *>(node.op);
    const MultiFeature *multi_feature = op->multi_feature();
    EXPECT_EQ(op->type(), multi_type);
    if (multi_type == MultiFeaOpType::kCrossFeatureOp) {
      const CrossFeature *cross_feature = dynamic_cast<const CrossFeature *>(multi_feature);
      EXPECT_NE(cross_feature, nullptr);
    } else if (multi_type == MultiFeaOpType::kMergeFeatureOp) {
      const MergeFeature *merge_feature = dynamic_cast<const MergeFeature *>(multi_feature);
      EXPECT_NE(merge_feature, nullptr);
    } else if (multi_type == MultiFeaOpType::kVectorFeatureOp) {
      const VectorFeature *vector_feature = dynamic_cast<const VectorFeature *>(multi_feature);
      EXPECT_NE(vector_feature, nullptr);
    } else {
      EXPECT_TRUE(false);
    }
  } else {
    EXPECT_TRUE(false);
  }
}

TEST(ExprParserTest, Default) {
  using xdl::io::ExprParser;
  using xdl::io::FeatureNameVec;

  ExprNode output_node;
  std::vector<ExprNode> internal_nodes;

  ExprParser expr_parser;
  expr_parser.Init(&internal_nodes);
  expr_parser.Parse(" sum(value(match (min(ad_cate_id), log( nick_cate_pv_14) )))",
                    output_node);

  EXPECT_EQ(internal_nodes.size(), 5);

  const FeatureNameVec &feature_name_vec = expr_parser.feature_name_vec();
  EXPECT_EQ(feature_name_vec.size(), 2);
  EXPECT_EQ(feature_name_vec[0], "ad_cate_id");
  EXPECT_EQ(feature_name_vec[1], "nick_cate_pv_14");

  printf("  Check(%s)\n", "sum");
  CheckExprNode(output_node, 1,
                FeaOpType::kSingleFeatureOp, true,
                SingleFeaOpType::kStatisFeatureOp, false);

  printf("  Check(%s)\n", "match");
  ExprNode &match_node = internal_nodes[output_node.pres[0]];
  CheckExprNode(match_node, 2,
                FeaOpType::kMultiFeatureOp, false,
                SingleFeaOpType::kDefaultSingleFeatureOp, false,
                MultiFeaOpType::kMergeFeatureOp);

  printf("  Check(%s)\n", "min");
  ExprNode &min_node = internal_nodes[match_node.pres[0]];
  CheckExprNode(min_node, 1,
                FeaOpType::kSingleFeatureOp, false,
                SingleFeaOpType::kStatisFeatureOp, false);

  printf("  Check(%s)\n", "ad_cate_id");
  ExprNode &source_node_0 = internal_nodes[min_node.pres[0]];
  CheckExprNode(source_node_0, 0,
                FeaOpType::kSourceFeatureOp, false);

  printf("  Check(%s)\n", "log");
  ExprNode &log_node = internal_nodes[match_node.pres[1]];
  CheckExprNode(log_node, 1,
                FeaOpType::kSingleFeatureOp, false,
                SingleFeaOpType::kTraversalFeatureOp, false);

  printf("  Check(%s)\n", "nick_cate_pv_14");
  ExprNode &source_node_1 = internal_nodes[log_node.pres[0]];
  CheckExprNode(source_node_1, 0,
                FeaOpType::kSourceFeatureOp, false);
}
