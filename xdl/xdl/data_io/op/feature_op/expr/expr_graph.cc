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


#include "xdl/data_io/op/feature_op/expr/expr_graph.h"

#include <xdl/core/utils/logging.h>

#include "xdl/core/lib/thread_local.h"
#include "xdl/data_io/op/feature_op/expr/dsl_parser.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_op.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_op.h"
#include "xdl/data_io/op/feature_op/source_feature_op/source_feature_op.h"
#include "xdl/proto/sample.pb.h"

namespace xdl {
namespace io {

ExprGraph::ExprGraph(const std::vector<std::string> &dsl_arr, bool is_cache) {
  is_cache_ = is_cache;
  dsl_parser_ = DslParser::Get();
  for (const std::string &dsl : dsl_arr)  dsl_parser_->Parse(dsl);
  expr_parser_.Init(internal_nodes_.mutable_nodes());
  ParseDsl(dsl_parser_->dsl_unit_map());
}

void ExprGraph::ParseDsl(const DslUnitMap &dsl_unit_map) {
  for (auto &iter : dsl_unit_map) {
    const DslUnit &dsl_unit = iter.second;
    ExprNode output_node;
    expr_parser_.Parse(dsl_unit.expr, output_node);
    output_nodes_map_.insert(std::make_pair(dsl_unit.name, internal_nodes_.nodes().size()));
    internal_nodes_.mutable_nodes()->push_back(output_node);
  }
}

void ExprGraph::Execute(const std::vector<const FeatureMap *> &feature_map_arr,
                        FeatureLine *feature_line, bool is_clear_result_feature) {
  using xdl::ThreadLocalStore;
  bool new_created;
  InternalNodes *local_internal_nodes = ThreadLocalStore<InternalNodes>::Get(&new_created, internal_nodes_);
  FeatureCacheMap *feature_cache = is_cache_ ? ThreadLocalStore<FeatureCacheMap>::Get(&new_created) : nullptr;
  if (feature_cache != nullptr && is_clear_result_feature) {
    for (auto &iter : *feature_cache)  iter.second.clear();
  }
  auto mutable_features = feature_line->mutable_features();
  mutable_features->Reserve(feature_line->features_size() + output_nodes_map_.size());
  for (const auto &iter : output_nodes_map_) {
    ExprNode &output_node = local_internal_nodes->node(iter.second);
    Feature *feature = reinterpret_cast<Feature *>(output_node.result);
    // TODO: 该output如果source缺的话就不用算了
    if (TraversalExecute(feature_map_arr, local_internal_nodes, output_node, feature_cache, is_clear_result_feature)) {
      if (!feature->has_name())  feature->set_name(iter.first);
      mutable_features->AddAllocated(feature);
      output_node.result = new Feature();
    }
  }
}

bool ExprGraph::TraversalExecute(const std::vector<const FeatureMap *> &feature_map_arr,
                                 InternalNodes *internal_nodes,
                                 ExprNode &node,
                                 FeatureCacheMap *feature_cache,
                                 bool is_clear_result_feature) {
  for (int pre : node.pres) {
    if (TraversalExecute(feature_map_arr, internal_nodes, internal_nodes->node(pre), feature_cache) == false) {
      return false;
    }
  }
  if (node.type == FeaOpType::kSourceFeatureOp) {
    XDL_CHECK(node.pres.size() == 0);
    if (!is_clear_result_feature && node.table_id > 0)  return true;
    SourceFeatureOp *op = reinterpret_cast<SourceFeatureOp *>(node.op);
    const auto &iter = feature_name_map_->find(op->name());
    XDL_CHECK(iter != feature_name_map_->end());
    XDL_CHECK(iter->second >= 0);
    if (iter->second >= feature_map_arr.size())  return false;
    if (op->Run(feature_map_arr[iter->second], node.result) == false)  return false;
    node.table_id = iter->second;
  } else {
    if (is_clear_result_feature || node.table_id == 0) {
      node.clear();
    } else {
      return true;
    }
    if (node.type == FeaOpType::kSingleFeatureOp) {
      XDL_CHECK(node.pres.size() == 1);
      SingleFeatureOp *op = reinterpret_cast<SingleFeatureOp *>(node.op);
      const ExprNode &pre_node = internal_nodes->node(node.pres[0]);
      if (op->Run(&pre_node, &node) == false)  return false;
      node.table_id = pre_node.table_id;
    } else if (node.type == FeaOpType::kMultiFeatureOp) {
      const size_t pres_size = node.pres.size();
      XDL_CHECK(pres_size > 1);
      for (size_t i = 0; i < pres_size; ++i) {
        const ExprNode &pre_node = internal_nodes->node(node.pres[i]);
        node.pre_nodes[i] = &pre_node;
        if (node.table_id < 0)  node.table_id = pre_node.table_id;
        else if (node.table_id > pre_node.table_id)  node.table_id = pre_node.table_id;
      }
      MultiFeatureOp *op = reinterpret_cast<MultiFeatureOp *>(node.op);
      if (op->Run(node.pre_nodes, &node) == false)  return false;
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace io
}  // namespace xdl
