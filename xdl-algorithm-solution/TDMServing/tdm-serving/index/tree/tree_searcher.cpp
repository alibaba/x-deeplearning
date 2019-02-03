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

#include "index/tree/tree_searcher.h"
#include <unordered_map>
#include "biz/filter.h"
#include "index/tree/tree_index_conf.h"
#include "index/tree/tree.h"
#include "index/tree/tree_search_context.h"
#include "util/str_util.h"
#include "util/log.h"

namespace tdm_serving {

bool CompareNodeScoreforParentWinner(const NodeScore* const node_1,
                                     const NodeScore* const node_2) {
  bool parent_1_win = node_1->parent() == NULL ||
      node_1->parent()->is_winner();
  bool parent_2_win = node_2->parent() == NULL ||
      node_2->parent()->is_winner();

  if (parent_1_win && !parent_2_win) {
    return true;
  } else if (!parent_1_win && parent_2_win) {
    return false;
  } else {
    return node_1->score() > node_2->score();
  }
}

TreeSearcher::TreeSearcher()
  : index_conf_(NULL) {
}

TreeSearcher::~TreeSearcher() {
}

bool TreeSearcher::Init(const TreeIndexConf* index_conf) {
  index_conf_ = index_conf;
  return true;
}

bool TreeSearcher::Search(Tree* tree,
                          TreeSearchContext* context,
                          const SearchParam& search_param) {
  // resize context node layer to avoid memory reallocation
  context->resize_node_layers(tree->max_level() + 1);

  // add root node, score reset to 1
  context->add_node_score(tree->root(), kRootLevel, 1.0);

  uint32_t level = kRootLevel;
  uint32_t max_level = tree->max_level();

  // BeamSearch
  bool is_break = false;
  bool ret = false;
  while (!is_break) {
    // calculate score
    ret = CalculateNodes(context, search_param, level, max_level);
    if (!ret) {
      return false;
    }

    // sort
    SortNodes(context, search_param, level, max_level);

    // end
    if (level == max_level) {
      break;
    }

    // spread nodes
    SpreadNodes(context, search_param, level);

     // next level
    level++;
  }

  return true;
}

bool TreeSearcher::CalculateNodes(TreeSearchContext* context,
                                  const SearchParam& search_param,
                                  uint32_t level, uint32_t max_level) {
  std::vector<NodeScore*> node_scores;

  NodeScoreVec* candidates = context->layers_node_scores(level);
  uint32_t candidate_size = context->layer_node_score_size(level);
  uint32_t ltopn = index_conf_->tree_level_topn(level);

  // do not calculate score for root node
  // do not calculate score for level which has fewer nodes, need not sort
  // do not calculate score for max level
  //     which has fewer nodes than search topn
  if (level == kRootLevel ||
      (level != max_level && candidate_size <= ltopn) ||
      (level == max_level && candidate_size <= search_param.topn())) {
    LOG_DEBUG << "level " << level <<
        " not need calc, " << candidate_size << " to " << ltopn;
    return true;
  }
  LOG_DEBUG << "level " << level <<
      " need calc, " << candidate_size << " to " << ltopn;

  // calculate score
  for (uint32_t i = 0; i < candidate_size; ++i) {
    NodeScore* node_score = candidates->at(i);

    // for item feature
    node_score->set_feature_group_id(
        &index_conf_->item_feature_group_id());

    // do not recalculate
    if (node_score->node_level() == level) {
      node_scores.push_back(node_score);
    } else {
      LOG_WARN << "node level mismatch, layer: " << level <<
          " != node level: " << node_score->node_level() <<
          " for node_id: " << node_score->item_id();
    }
  }

  // batch process
  std::vector<ItemFeature*> item_features(node_scores.begin(),
                                          node_scores.end());
  if (!CalculateScore(context, search_param,
                      &item_features, &node_scores)) {
    return false;
  }

  return true;
}

void TreeSearcher::SortNodes(TreeSearchContext* context,
                             const SearchParam& search_param,
                             uint32_t level, uint32_t max_level) {
  NodeScoreVec* candidates = context->layers_node_scores(level);
  uint32_t candidate_size = context->layer_node_score_size(level);
  uint32_t ltopn = index_conf_->tree_level_topn(level);

  // sort
  if ((level != max_level && candidate_size > ltopn) ||
      (level == max_level && candidate_size > search_param.topn())) {
    uint32_t sortn = ltopn;
    if (level == max_level) {
      sortn = candidate_size;
    }

    LOG_DEBUG << "level " << level <<
        " need sort, " << candidate_size << " to " << sortn;

    std::partial_sort(candidates->begin(),
                      candidates->begin() + sortn,
                      candidates->begin() + candidate_size,
                      CompareNodeScoreforParentWinner);

    // get winner, candidate is winner by default, update here
    for (uint32_t i = sortn; i < candidate_size; i++) {
      candidates->at(i)->set_is_winner(false);
    }

    // if the winner is a leaf, put to last layer
    if (level != max_level) {
      for (uint32_t i = 0; i < sortn; i++) {
        NodeScore* node_score = candidates->at(i);
        Node* node = node_score->node();

        uint32_t childs_size = node->sub_node_size();
        if (childs_size != 0) {
          continue;
        }

        if (context->filter() != NULL) {
          NodeScore item(node);
          if (context->filter()->IsFiltered(&search_param.filter_info(),
                                            item)) {
            LOG_DEBUG << "node: " << node->node_info()->id() <<
                " with cat_id: " << node->node_info()->leaf_cate_id() <<
                " is bought filtered";
            continue;
          }
        }

        context->add_node_score(node, max_level, node_score->score());
      }
    }
  } else {
    LOG_DEBUG << "level " << level <<
        " not need sort, " << candidate_size << " to " << ltopn;
  }
}

void TreeSearcher::SpreadNodes(TreeSearchContext* context,
                               const SearchParam& search_param,
                               uint32_t level) {
  NodeScoreVec* candidates = context->layers_node_scores(level);
  uint32_t candidate_size = context->layer_node_score_size(level);

  for (uint32_t i = 0; i < candidate_size; ++i) {
    NodeScore* node_score = candidates->at(i);
    Node* node = node_score->node();

    // filter by sort result
    // sort result is topn winers and subsequent candidate nodes
    if (!node_score->is_winner()) {
      break;
    }

    // add child
    uint32_t childs_size = node->sub_node_size();
    for (size_t j = 0; j < childs_size; j++) {
      Node* sub_node = node->sub_node(j);
      if (sub_node == NULL) {
        continue;
      }

      if (context->filter() != NULL) {
        NodeScore item(sub_node);
        if (context->filter()->IsFiltered(&search_param.filter_info(), item)) {
          LOG_DEBUG << "node: " << sub_node->node_info()->id()
                    << " is filtered";
          continue;
        }
      }

      context->add_node_score(sub_node,
          sub_node->node_info()->level(), 1, node_score);
    }
  }
}

bool TreeSearcher::CalculateScore(TreeSearchContext* context,
                                  const SearchParam& search_param,
                                  std::vector<ItemFeature*>* item_features,
                                  std::vector<NodeScore*>* node_scores) {
  PredictRequest predict_req;
  PredictResponse predict_res;

  predict_req.set_model_name(index_conf_->model_name());
  predict_req.set_model_version(index_conf_->model_version());
  predict_req.set_item_features(item_features);

  if (search_param.has_user_info() &&
      search_param.user_info().has_user_feature()) {
    predict_req.set_user_info(&search_param.user_info());
  } else {
    predict_req.set_user_info(NULL);
  }

  bool ret = ModelManager::Instance().Predict(
      context->mutable_predict_context(),
      predict_req, &predict_res);
  if (!ret) {
    LOG_ERROR << "model predict failed.";
    return false;
  }

  for (size_t i = 0; i < predict_res.score_size(); i++) {
    node_scores->at(i)->set_score(predict_res.score(i));
  }

  return true;
}

}  // namespace tdm_serving
