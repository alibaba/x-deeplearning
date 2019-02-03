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

#ifndef TDM_SERVING_INDEX_TREE_TREE_SEARCH_CONTEXT_H_
#define TDM_SERVING_INDEX_TREE_TREE_SEARCH_CONTEXT_H_

#include <tr1/unordered_map>
#include "index/search_context.h"
#include "index/item.h"
#include "index/tree/tree_def.h"
#include "index/tree/node.h"
#include "model/predict_interface.h"

namespace tdm_serving {

// Tree node info used in tree searching
class NodeScore : public Item, public ItemFeature {
 public:
  NodeScore();
  explicit NodeScore(Node* node);
  NodeScore(Node* node, float score);
  NodeScore(Node* node, float score, NodeScore* parent);

  void set_node(Node* node) {
    node_ = node;
  }

  Node* node() {
    return node_;
  }

  void set_parent(NodeScore* parent) {
    parent_ = parent;
  }

  NodeScore* parent() const {
    return parent_;
  }

  void set_is_winner(bool is_winner) {
    is_winner_ = is_winner;
  }

  bool is_winner() const {
    return is_winner_;
  }

  uint32_t node_level() const {
    return node_->node_info()->level();
  }

  // item interface
  // used for filter and sort item
  uint64_t item_id() const {
    return node_->node_info()->id();
  }

  void set_score(float score) {
    score_ = score;
  }

  float score() const {
    return score_;
  }

  bool has_category() const {
    return node_->node_info()->has_leaf_cate_id();
  }

  int32_t category() const {
    return node_->node_info()->leaf_cate_id();
  }

  // item feature interface
  // used in model layer
  virtual size_t feature_group_size() const {
    return 1;
  }

  virtual const std::string& feature_group_id(size_t /*grp_index*/) const {
    return *feature_group_id_;
  }

  virtual size_t feature_entity_size(size_t /*grp_index*/) const {
    return 1;
  }

  virtual uint64_t feature_entity_id(size_t /*grp_index*/,
                                     size_t /*ent_index*/) const {
    return node_->node_info()->hashid();
  }

  virtual float feature_entity_value(size_t /*grp_index*/,
                                     size_t /*ent_index*/) const {
    return 1;
  }

  void set_feature_group_id(const std::string* feature_group_id) {
    feature_group_id_ = feature_group_id;
  }

 private:
  // ref to tree node
  Node* node_;

  // parent
  NodeScore* parent_;

  // calculated score
  float score_;

  // if this node is winer after sorting
  // set true for default
  bool is_winner_;

  // feature group id, used for item feature interface
  // tree item feature can be of one feature group now 
  const std::string* feature_group_id_;
};

typedef std::vector<NodeScore*> NodeScoreVec;
typedef std::vector<NodeScoreVec> NodeLayers;

// Search context used for tree searching
class TreeSearchContext : public SearchContext {
 public:
  TreeSearchContext()
    : node_layer_size_(0) {}

  virtual ~TreeSearchContext();

  virtual void Clear();

  // Add node candidate
  NodeScore* add_node_score(Node* node, uint32_t level, float score) {
    NodeScoreVec* node_score_vec = layers_node_scores(level);

    node_score_size_[level]++;
    if (node_score_size_[level] > node_score_vec->size()) {
      node_score_vec->push_back(new NodeScore());
    }
    NodeScore* node_score = node_score_vec->at(node_score_size_[level] - 1);

    node_score->set_node(node);
    node_score->set_score(score);
    node_score->set_parent(NULL);
    node_score->set_is_winner(true);

    return node_score;
  }

  // Add node candidate info with parent
  NodeScore* add_node_score(Node* node, uint32_t level,
                            float score, NodeScore* parent) {
    NodeScoreVec* node_score_vec = layers_node_scores(level);

    node_score_size_[level]++;
    if (node_score_size_[level] > node_score_vec->size()) {
      node_score_vec->push_back(new NodeScore());
    }
    NodeScore* node_score = node_score_vec->at(node_score_size_[level] - 1);

    node_score->set_node(node);
    node_score->set_score(score);
    node_score->set_parent(parent);
    node_score->set_is_winner(true);

    return node_score;
  }

  // Get candidate size by tree search level
  uint32_t layer_node_score_size(uint32_t level) {
    return node_score_size_[level];
  }

  // Get candidates by tree search level
  NodeScoreVec* layers_node_scores(uint32_t level) {
    if (level > node_layer_size_ - 1) {
      resize_node_layers(level + 1);
    }
    return &node_layers_[level];
  }

  // Get candidate size of last tree search level
  uint32_t last_layer_node_score_size() {
    if (node_layer_size_ > 0) {
      return node_score_size_[node_layer_size_ - 1];
    } else {
      return 0;
    }
  }

  // Get candidates of last tree search level
  NodeScoreVec* last_layer_node_scores() {
    if (node_layer_size_ > 0) {
      return &node_layers_[node_layer_size_ - 1];
    } else {
      return NULL;
    }
  }

  // Resize candidate layers
  void resize_node_layers(uint32_t size) {
    for (uint32_t i = node_layers_.size(); i < size; i++) {
      node_layers_.push_back(NodeScoreVec());
      node_score_size_.push_back(0);
    }
    LOG_DEBUG << "layer_size: " << node_layers_.size();
    LOG_DEBUG << "score_size: " << node_score_size_.size();
    node_layer_size_ = size;
  }

 private:
  // candidate layers
  NodeLayers node_layers_;

  // candidate layer size
  uint32_t node_layer_size_;

  // candidate size of each layer
  std::vector<uint32_t> node_score_size_;

  DISALLOW_COPY_AND_ASSIGN(TreeSearchContext);
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_TREE_TREE_SEARCH_CONTEXT_H_
