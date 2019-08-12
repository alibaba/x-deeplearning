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
#include "xdl/data_io/op/feature_op/feature_op_type.h"

namespace xdl {
namespace io {

class ExprParser {
 public:
  void Init(std::vector<ExprNode> *internal_nodes) {
    internal_nodes_ = internal_nodes;
  }

  void Parse(const std::string &expr, ExprNode &node, bool is_output = true);

  const FeatureNameVec &feature_name_vec() const { return feature_name_vec_; }

 protected:
  void ParseNode(const std::string &expr,
                 std::vector<std::string> &sub_exprs, ExprNode &node,
                 bool is_output);
  void InsertFeatureNameSets(const std::string &feature_name);

 private:
  FeatureNameVec feature_name_vec_;
  std::vector<ExprNode> *internal_nodes_;
};

}  // namespace io
}  // namespace xdl