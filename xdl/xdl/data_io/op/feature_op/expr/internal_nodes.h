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

#include <vector>

#include "xdl/data_io/op/feature_op/expr/expr_node.h"
#include "xdl/data_io/op/feature_op/expr/internal_feature.h"

namespace xdl {
namespace io {

class InternalNodes {
 public:
  InternalNodes() = default;
  explicit InternalNodes(const InternalNodes &internal_nodes) {
    for (const ExprNode &node : internal_nodes.nodes()) {
      nodes_.push_back(node);  // copy
      nodes_.back().InitResult();
    }
  }

  virtual ~InternalNodes() {
    for (ExprNode &node : nodes_) {
      node.ReleaseResult();
      node.ReleaseOp();
    }
  }

  const std::vector<ExprNode> &nodes() const { return nodes_; }
  std::vector<ExprNode> *mutable_nodes() { return &nodes_; }
  ExprNode &node(int index) { return nodes_[index]; }

 private:
  std::vector<ExprNode> nodes_;
};

}  // namespace io
}  // namespace xdl