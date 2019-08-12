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

#include "xdl/data_io/op/feature_op/expr/expr_node.h"
#include "xdl/data_io/op/feature_op/expr/expr_parser.h"
#include "xdl/data_io/op/feature_op/expr/internal_nodes.h"
#include "xdl/data_io/op/feature_op/feature_op_type.h"

namespace xdl {
namespace io {

class ExprGraph {
 public:
  virtual ~ExprGraph() = default;

  static ExprGraph *Get(const std::vector<std::string> &dsl_arr, bool is_cache = false) {
    static ExprGraph expr_graph(dsl_arr, is_cache);
    return &expr_graph;
  }

  void Execute(const std::vector<const FeatureMap *> &feature_map_arr,
               FeatureLine *feature_line, bool is_clear_result_feature = true);

  const FeatureNameVec &feature_name_vec() const { return expr_parser_.feature_name_vec(); }
  inline void set_feature_name_map(const FeatureNameMap *feature_name_map) {
    feature_name_map_ = feature_name_map;
  }

 protected:
  ExprGraph(const std::vector<std::string> &dsl_arr, bool is_cache = false);

  void ParseDsl(const DslUnitMap &dsl_unit_map);
  bool TraversalExecute(const std::vector<const FeatureMap *> &feature_map_arr,
                        InternalNodes *internal_nodes,
                        ExprNode &node,
                        FeatureCacheMap *feature_cache = nullptr,
                        bool is_clear_result_feature = true);

 private:
  InternalNodes internal_nodes_;
  std::unordered_map<std::string, int> output_nodes_map_;
  const FeatureNameMap *feature_name_map_;
  ExprParser expr_parser_;
  DslParser *dsl_parser_ = nullptr;
  bool is_cache_ = false;
};

}  // namespace io
}  // namespace xdl