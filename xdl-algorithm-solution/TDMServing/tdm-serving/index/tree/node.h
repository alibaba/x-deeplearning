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

#ifndef TDM_SERVING_INDEX_TREE_NODE_H_
#define TDM_SERVING_INDEX_TREE_NODE_H_

#include "common/common_def.h"
#include "proto/tree.pb.h"

namespace tdm_serving {

class Tree;

class Node {
 public:
  Node();
  ~Node();

  bool InitNodeStructure(Tree* tree);

  UINode* node_info() {
    return node_info_;
  }

  void set_node_info(UINode* node_info) {
    node_info_ = node_info;
  }

  Node* parent() {
    return parent_node_;
  }

  uint32_t sub_node_size() {
    return sub_node_num_;
  }

  Node* sub_node(uint32_t index) {
    return sub_nodes_[index];
  }

  void sub_nodes(Node** sub_nodes, uint32_t* sub_node_num) {
    *sub_nodes = *(sub_nodes_);
    *sub_node_num = sub_node_num_;
  }

 private:
  UINode* node_info_;

  Node** sub_nodes_;
  uint32_t sub_node_num_;
  Node* parent_node_;

  DISALLOW_COPY_AND_ASSIGN(Node);
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_TREE_NODE_H_
