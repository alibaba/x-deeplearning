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

#include "index/tree/tree_search_context.h"
#include <stdio.h>
#include "util/str_util.h"
#include "util/log.h"

namespace tdm_serving {

// ----------------------- NodeScore -----------------------
NodeScore::NodeScore()
    : node_(NULL), parent_(NULL), score_(0), is_winner_(true) {
}

NodeScore::NodeScore(Node* node)
    : node_(node), parent_(NULL), score_(0), is_winner_(true) {
}

NodeScore::NodeScore(Node* node, float score)
    : node_(node), parent_(NULL), score_(score), is_winner_(true) {
}

NodeScore::NodeScore(Node* node, float score, NodeScore* parent)
    : node_(node), parent_(parent), score_(score), is_winner_(true) {
}

// ----------------------- TreeSearchContext -----------------------
TreeSearchContext::~TreeSearchContext() {
  for (uint32_t i = 0; i < node_layers_.size(); i++) {
    for (uint32_t j = 0; j < node_layers_[i].size(); j++) {
      delete node_layers_[i][j];
    }
  }
}

void TreeSearchContext::Clear() {
  for (uint32_t i = 0; i < node_layer_size_; ++i) {
    node_score_size_[i] = 0;
  }
  node_layer_size_ = 0;

  SearchContext::Clear();
}

}  // namespace tdm_serving
