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

#define protected public
#define private public

#include "index/tree/tree_searcher.h"
#include "index/tree/tree_index_conf.h"
#include "index/tree/tree_search_context.h"
#include "index/tree/node.h"
#include "index/tree/tree.h"
#include "biz/filter.h"
#include "util/log.h"
#include "proto/search.pb.h"

namespace tdm_serving {

class MockTreeSearcher : public TreeSearcher {
 protected:
  virtual bool CalculateScore(TreeSearchContext* /*context*/,
                              const SearchParam& /*search_param*/,
                              std::vector<ItemFeature*>* /*item_features*/,
                              std::vector<NodeScore*>* node_scores) {
    for (size_t i = 0; i < node_scores->size(); i++) {
      double score = node_scores->at(i)->node()->node_info()->id();
      if (score != 0) {
        score = 1 / score;
      }
      node_scores->at(i)->set_score(score);
    }
    return true;
  }
};

class TreeMockFilter : public Filter {
 public:
  virtual bool IsFiltered(const FilterInfo* /*filter_info*/,
                          const Item& item) {
    if (item.item_id() == 5 ||
        item.item_id() == 9 ||
        item.item_id() == 10) {
      LOG_INFO << "item: " << item.item_id()
               << "is filtered";
      return true;
    }
    return false;
  }
};

/*
 * search tree:
 *        0
 *      /   \
 *     1     2
 *   /  \   /  \
 *  6   3   4    5
 *     / \ / \
 *    7  8 9  10
 * beam seach, topk = 3 
 * sort by node num
 * expected result: 5, 7, 8, 9, 10
*/

void MockTree(Tree* tree) {
  uint32_t max_level = 3;
  uint32_t total_node_num = 11;

  tree->tree_meta_.max_level_ = max_level;
  tree->tree_meta_.total_node_num_ = total_node_num;

  tree->nodes_ = new Node[total_node_num];
  for (uint32_t i = 0; i < total_node_num; i++) {
    tree->nodes_[i].node_info_ = new UINode;
  }

  Node* tns = tree->nodes_;

  tns[0].node_info_->set_id(0);
  tns[0].node_info_->set_level(0);
  tns[0].sub_node_num_ = 2;
  tns[0].sub_nodes_ = new Node*[2];
  tns[0].sub_nodes_[0] = tns + 1;
  tns[0].sub_nodes_[1] = tns + 2;

  tns[1].node_info_->set_id(1);
  tns[1].node_info_->set_level(1);
  tns[1].sub_node_num_ = 2;
  tns[1].sub_nodes_ = new Node*[2];
  tns[1].sub_nodes_[0] = tns + 3;
  tns[1].sub_nodes_[1] = tns + 4;

  tns[2].node_info_->set_id(2);
  tns[2].node_info_->set_level(1);
  tns[2].sub_node_num_ = 0;
  tns[2].sub_node_num_ = 2;
  tns[2].sub_nodes_ = new Node*[2];
  tns[2].sub_nodes_[0] = tns + 5;
  tns[2].sub_nodes_[1] = tns + 6;

  tns[3].node_info_->set_id(6);
  tns[3].node_info_->set_level(2);
  tns[3].sub_node_num_ = 0;

  tns[4].node_info_->set_id(3);
  tns[4].node_info_->set_level(2);
  tns[4].sub_node_num_ = 2;
  tns[4].sub_nodes_ = new Node*[2];
  tns[4].sub_nodes_[0] = tns + 7;
  tns[4].sub_nodes_[1] = tns + 8;

  tns[5].node_info_->set_id(4);
  tns[5].node_info_->set_level(2);
  tns[5].sub_node_num_ = 2;
  tns[5].sub_nodes_ = new Node*[2];
  tns[5].sub_nodes_[0] = tns + 9;
  tns[5].sub_nodes_[1] = tns + 10;

  tns[6].node_info_->set_id(5);
  tns[6].node_info_->set_level(2);
  tns[6].node_info_->set_leaf_cate_id(32);
  tns[6].sub_node_num_ = 0;

  tns[7].node_info_->set_id(7);
  tns[7].node_info_->set_level(3);
  tns[7].sub_node_num_ = 0;

  tns[8].node_info_->set_id(8);
  tns[8].node_info_->set_level(3);
  tns[8].sub_node_num_ = 0;

  tns[9].node_info_->set_id(9);
  tns[9].node_info_->set_level(3);
  tns[9].node_info_->set_leaf_cate_id(8);
  tns[9].sub_node_num_ = 0;

  tns[10].node_info_->set_id(10);
  tns[10].node_info_->set_level(3);
  tns[10].node_info_->set_leaf_cate_id(16);
  tns[10].sub_node_num_ = 0;
}

TEST(TreeSearcher, beam_search) {
  MockTreeSearcher s;

  TreeIndexConf index_conf;
  index_conf.level_to_topn_[0] = 3;
  index_conf.level_to_topn_[1] = 3;
  index_conf.level_to_topn_[2] = 3;
  index_conf.level_to_topn_[3] = 3;
  s.index_conf_ = &index_conf;

  Tree tree;
  MockTree(&tree);

  TreeSearchContext context;
  SearchParam search_param;

  ASSERT_TRUE(s.Search(&tree, &context, search_param));

  NodeScoreVec* candidates =
      context.layers_node_scores(tree.tree_meta_.max_level_);

  ASSERT_EQ(5u, context.layer_node_score_size(tree.tree_meta_.max_level_));
  EXPECT_EQ(5u, candidates->at(0)->node()->node_info()->id());
  EXPECT_EQ(7u, candidates->at(1)->node()->node_info()->id());
  EXPECT_EQ(8u, candidates->at(2)->node()->node_info()->id());
  EXPECT_EQ(9u, candidates->at(3)->node()->node_info()->id());
  EXPECT_EQ(10u, candidates->at(4)->node()->node_info()->id());
}

TEST(TreeSearcher, beam_search_batch_process) {
  MockTreeSearcher s;

  TreeIndexConf index_conf;
  index_conf.model_batch_num_ = 3;
  index_conf.level_to_topn_[0] = 3;
  index_conf.level_to_topn_[1] = 3;
  index_conf.level_to_topn_[2] = 3;
  index_conf.level_to_topn_[3] = 3;
  s.index_conf_ = &index_conf;

  Tree tree;
  MockTree(&tree);

  TreeSearchContext context;
  SearchParam search_param;

  ASSERT_TRUE(s.Search(&tree, &context, search_param));

  NodeScoreVec* candidates =
      context.layers_node_scores(tree.tree_meta_.max_level_);

  ASSERT_EQ(5u, context.layer_node_score_size(tree.tree_meta_.max_level_));
  EXPECT_EQ(5u, candidates->at(0)->node()->node_info()->id());
  EXPECT_EQ(7u, candidates->at(1)->node()->node_info()->id());
  EXPECT_EQ(8u, candidates->at(2)->node()->node_info()->id());
  EXPECT_EQ(9u, candidates->at(3)->node()->node_info()->id());
  EXPECT_EQ(10u, candidates->at(4)->node()->node_info()->id());
}

TEST(TreeSearcher, beam_search_with_filter) {
  MockTreeSearcher s;

  TreeIndexConf index_conf;
  index_conf.level_to_topn_[0] = 3;
  index_conf.level_to_topn_[1] = 3;
  index_conf.level_to_topn_[2] = 3;
  index_conf.level_to_topn_[3] = 3;
  s.index_conf_ = &index_conf;

  Tree tree;
  MockTree(&tree);

  TreeSearchContext context;

  Filter* filter = new TreeMockFilter;
  context.set_filter(filter);

  SearchParam search_param;

  ASSERT_TRUE(s.Search(&tree, &context, search_param));

  NodeScoreVec* candidates =
      context.layers_node_scores(tree.tree_meta_.max_level_);

  ASSERT_EQ(2u, context.layer_node_score_size(tree.tree_meta_.max_level_));
  EXPECT_EQ(7u, candidates->at(0)->node()->node_info()->id());
  EXPECT_EQ(8u, candidates->at(1)->node()->node_info()->id());
}

TEST(TreeSearcher, beam_search_level_topn) {
  MockTreeSearcher s;

  TreeIndexConf index_conf;
  index_conf.level_to_topn_[0] = 3;
  index_conf.level_to_topn_[1] = 1;
  index_conf.level_to_topn_[2] = 1;
  index_conf.level_to_topn_[3] = 3;
  s.index_conf_ = &index_conf;

  Tree tree;
  MockTree(&tree);

  TreeSearchContext context;
  SearchParam search_param;

  ASSERT_TRUE(s.Search(&tree, &context, search_param));

  NodeScoreVec* candidates =
      context.layers_node_scores(tree.tree_meta_.max_level_);

  ASSERT_EQ(2u, context.layer_node_score_size(tree.tree_meta_.max_level_));
  EXPECT_EQ(7u, candidates->at(0)->node()->node_info()->id());
  EXPECT_EQ(8u, candidates->at(1)->node()->node_info()->id());
}

}  // namespace tdm_serving
