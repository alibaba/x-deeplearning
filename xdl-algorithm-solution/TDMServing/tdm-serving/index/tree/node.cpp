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

#include "index/tree/node.h"
#include "index/tree/tree.h"
#include "util/log.h"

namespace tdm_serving {

Node::Node()
    : node_info_(NULL),
    sub_nodes_(NULL),
    sub_node_num_(0),
    parent_node_(NULL) {
}

Node::~Node() {
  DELETE_ARRAY(sub_nodes_);
}

bool Node::InitNodeStructure(Tree* tree) {
  sub_node_num_ = node_info_->children_size();
  sub_nodes_ = new Node*[sub_node_num_];
  for (uint32_t i = 0; i < sub_node_num_; ++i) {
    *(sub_nodes_ + i) = tree->node_by_seq(node_info_->children(i));
    if (*(sub_nodes_ + i) == NULL) {
      LOG_ERROR << "Get child node: "
                     << node_info_->children(i) << " failed";
      return false;
    }
  }

  if (node_info_->has_parent()) {
    parent_node_ = tree->node_by_seq(node_info_->parent());
    if (parent_node_ == NULL) {
      LOG_ERROR << "Get parent node: "
                     << node_info_->parent() << " failed";
      return false;
    }
  }
  return true;
}

}  // namespace tdm_serving
