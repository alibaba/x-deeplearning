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

#include "test/index/tree/tree_test_util.h"
#include "index/tree/tree.h"
#include "index/tree/tree_index_conf.h"

namespace tdm_serving {

TEST(Tree, Good) {
  // prepare index data
  std::string index_path = "test_data/index/tree_data";
  std::string index_data_path = index_path + "/pb_data_good";
  ASSERT_TRUE(CreateTestTreeIndex(index_data_path));

  // prepare index config
  TreeIndexConf index_conf;
  index_conf.section_ = "tree_index";
  index_conf.index_path_ = index_path;
  index_conf.latest_index_path_ = index_data_path;
  index_conf.version_file_path_ = index_path + "/version";
  index_conf.index_version_ = "20180419";
  index_conf.build_omp_ = 2;

  // Init tree
  Tree tree;
  ASSERT_TRUE(tree.Init(&index_conf));
  ASSERT_EQ(tree.total_node_num(), 15u);
  ASSERT_EQ(tree.max_level(), 3u);

  ASSERT_EQ(tree.root()->node_info()->seq(), 0);
  ASSERT_EQ(tree.root()->node_info()->id(), 1u);

  for (uint32_t i = 0; i < tree.total_node_num(); ++i) {
    // check node info
    Node* node = tree.node_by_seq(i);
    ASSERT_TRUE(node != NULL);
    ASSERT_EQ(node->node_info()->seq(), static_cast<int32_t>(i));
    ASSERT_EQ(node->node_info()->id(), GetIdBySeq(i));
    ASSERT_EQ(node, tree.node_by_id(GetIdBySeq(i)));

    // check sub node
    uint32_t real_sub_node_num = 0;
    if ((i * 2 + 1) < tree.total_node_num()) {
      real_sub_node_num++;
    }
    if ((i * 2 + 2) < tree.total_node_num()) {
      real_sub_node_num++;
    }
    ASSERT_EQ(node->sub_node_size(), real_sub_node_num);

    uint32_t sub_node_num = 0;
    Node* sub_nodes = NULL;
    node->sub_nodes(&sub_nodes, &sub_node_num);
    ASSERT_EQ(real_sub_node_num, sub_node_num);
    ASSERT_TRUE(sub_nodes != NULL);
    for (uint32_t j = 0; j < sub_node_num; ++j) {
      Node* sub_node = sub_nodes + j;
      ASSERT_EQ(sub_node->node_info()->seq(), i * 2 + j + 1);
      ASSERT_EQ(sub_node, node->sub_node(j));
    }

    // check parent
    if (i == 0) {
      ASSERT_TRUE(node->parent() == NULL);
    } else {
      ASSERT_TRUE(node->parent() != NULL);
      ASSERT_EQ(node->parent()->node_info()->seq(), (i - 1) /2);
    }
  }

  // check not exits
  ASSERT_TRUE(tree.node_by_seq(16) == NULL);
  ASSERT_TRUE(tree.node_by_id(161) == NULL);
}

TEST(Tree, Bad) {
  // prepare index config
  std::string index_path = "test_data/index/tree_data";
  std::string index_data_path = index_path + "/pb_data_bad";

  TreeIndexConf index_conf;
  index_conf.section_ = "tree_index";
  index_conf.index_path_ = index_path;
  index_conf.latest_index_path_ = index_data_path;
  index_conf.version_file_path_ = index_path + "/version";
  index_conf.index_version_ = "20180419";
  index_conf.build_omp_ = 2;

  // meta pb parse failed
  {
    ASSERT_TRUE(CreateTestTreeIndex(index_data_path, true));
    Tree tree;
    ASSERT_FALSE(tree.Init(&index_conf));
  }
  // meta pb file not exits
  {
    ASSERT_TRUE(CreateTestTreeIndex(index_data_path, false, true));
    Tree tree;
    ASSERT_FALSE(tree.Init(&index_conf));
  }
  // empty tree
  {
    ASSERT_TRUE(CreateTestTreeIndex(index_data_path, false, false, true));
    Tree tree;
    ASSERT_FALSE(tree.Init(&index_conf));
  }
  // build_one_tree_pb_with_other_pb_caurse_parse_failed
  {
    ASSERT_TRUE(CreateTestTreeIndex(index_data_path, false, false,
                                    false, true));
    Tree tree;
    ASSERT_FALSE(tree.Init(&index_conf));
  }
  // build_one_tree_pb_file_not_exits_caurse_parse_failed
  {
    ASSERT_TRUE(CreateTestTreeIndex(index_data_path, false, false,
                                    false, false, true));
    Tree tree;
    ASSERT_FALSE(tree.Init(&index_conf));
  }
  // build_meta_offset_count_not_alligned
  {
    ASSERT_TRUE(CreateTestTreeIndex(
        index_data_path, false, false, false, false, false, true));
    Tree tree;
    ASSERT_FALSE(tree.Init(&index_conf));
  }
  // build_init_node_structure_child_failed
  {
    ASSERT_TRUE(CreateTestTreeIndex(
        index_data_path, false, false, false, false, false, false, true));
    Tree tree;
    ASSERT_FALSE(tree.Init(&index_conf));
  }
  // build_init_node_structure_parent_failed
  {
    ASSERT_TRUE(CreateTestTreeIndex(
        index_data_path,
        false, false, false, false, false, false, false, true));
    Tree tree;
    ASSERT_FALSE(tree.Init(&index_conf));
  }
  // build_duplicate_node
  {
    ASSERT_TRUE(CreateTestTreeIndex(
        index_data_path,
        false, false, false, false, false, false, false, false, true));
    Tree tree;
    ASSERT_FALSE(tree.Init(&index_conf));
  }
  // build_tree_node_size_not_equal_meta
  {
    ASSERT_TRUE(CreateTestTreeIndex(
        index_data_path,
        false, false, false, false, false, false, false, false, false, true));
    Tree tree;
    ASSERT_FALSE(tree.Init(&index_conf));
  }
}

}  // namespace tdm_serving
