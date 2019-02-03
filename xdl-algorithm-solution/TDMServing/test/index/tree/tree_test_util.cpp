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
#include "util/log.h"

namespace tdm_serving {

uint32_t GetIdBySeq(uint32_t seq) {
  return seq * 10 + 1;
}

uint32_t GetLevel(uint32_t id) {
  uint32_t cur_level = 0;
  uint32_t cur_level_max_id = 0;
  while (true) {
    if (id <= cur_level_max_id) {
      return cur_level;
    }
    cur_level++;
    cur_level_max_id =  cur_level_max_id * 2 + 2;
  }
}

// 四层完全二叉树，拆成三个子树
// 第一个子树: 11, 12, 13, 14
// 第二个子树: 3, 4, 5, 6, 7, 8, 9, 10
// 第三个子树: 0, 1, 2
bool CreateTestTreeIndex(
    const std::string& data_path,
    bool build_meta_pb_with_other_pb_caurse_failed,
    bool build_meta_pb_file_not_exits_caurse_parse_failed,
    bool build_empty_tree,
    bool build_one_tree_pb_with_other_pb_caurse_parse_failed,
    bool build_one_tree_pb_file_not_exits_caurse_parse_failed,
    bool build_meta_offset_count_not_alligned,
    bool build_init_node_structure_child_failed,
    bool build_init_node_structure_parent_failed,
    bool build_duplicate_node,
    bool build_tree_node_size_not_equal_meta) {
  uint32_t tree_num = 3;
  uint32_t total_node_num = 15;
  if (build_empty_tree) {
    tree_num = 0;
    total_node_num = 0;
  }

  std::vector<uint32_t> offsets;
  offsets.push_back(11);
  offsets.push_back(3);
  offsets.push_back(0);

  // build trees
  std::vector<UITree*> trees;
  for (uint32_t i = 0; i < tree_num; ++i) {
    trees.push_back(new UITree());
  }

  UITree* tree = NULL;;
  for (uint32_t i = 0; i < total_node_num; ++i) {
    for (uint32_t j = 0; j < offsets.size(); ++j) {
      if (offsets[j] == i) {
        tree = trees[j];
        break;
      }
    }
    UINode* node = tree->add_nodes();
    node->set_seq(i);
    node->set_id(GetIdBySeq(i));
    node->set_leaf_cate_id(GetIdBySeq(i));
    if (i == 1 && build_duplicate_node) {
      node->set_id(GetIdBySeq(i - 1));
    } else {
      node->set_id(GetIdBySeq(i));
    }
    node->set_level(GetLevel(i));
    if (i != 0) {
      if (!build_init_node_structure_parent_failed) {
        node->set_parent((i - 1) /2);
      } else {
        node->set_parent(10000);
      }
    }
    if ((i * 2 + 1) < total_node_num) {
      if (build_init_node_structure_child_failed) {
        node->add_children(10000);
      } else {
        node->add_children(i * 2 + 1);
      }
    }
    if ((i * 2 + 2) < total_node_num) {
      node->add_children(i * 2 + 2);
    }
  }

  // build meta
  UIMeta meta;
  for (uint32_t i = 0; i < trees.size(); ++i) {
    UITree* tree = trees[i];
    UIHead* head = meta.add_heads();
    head->set_tid(i);
    head->set_offset(offsets[i]);
    head->set_count(tree->nodes_size());
  }

  // export
  std::string command = "mkdir -p " + data_path;
  if (system(command.c_str()) != 0) {
    LOG_ERROR << "execute " << command << " failed";
    return false;
  }

  if (build_tree_node_size_not_equal_meta) {
    meta.mutable_heads(0)->set_count(meta.heads(0).count() - 1);
  }
  if (build_meta_offset_count_not_alligned) {
    meta.mutable_heads(2)->set_count(meta.heads(2).count() - 1);
  }

  std::string meta_path = data_path + "/meta.dat";
  if (build_meta_pb_with_other_pb_caurse_failed) {
    if (!ConvertPbToBinaryFile(meta_path, *(trees[1]))) {
      return false;
    }
  } else if (build_meta_pb_file_not_exits_caurse_parse_failed) {
    std::string command = "rm -rf " + meta_path;
    if (system(command.c_str()) != 0) {
      LOG_ERROR << "execute " << command << " failed";
      return false;
    }
  } else {
    if (!ConvertPbToBinaryFile(meta_path, meta)) {
      return false;
    }
  }

  if (build_tree_node_size_not_equal_meta) {
    meta.mutable_heads(0)->set_count(meta.heads(0).count() + 1);
  }
  if (build_meta_offset_count_not_alligned) {
    meta.mutable_heads(2)->set_count(meta.heads(2).count() + 1);
  }

  char tree_path[1024];
  for (uint32_t i = 0; i < trees.size(); ++i) {
    snprintf(tree_path, sizeof(tree_path), "%s/%s%u", data_path.c_str(),
        "tree.dat.", meta.heads(i).tid());
    if (i == 0 && build_one_tree_pb_with_other_pb_caurse_parse_failed) {
      if (!ConvertPbToBinaryFile(tree_path, meta)) {
        return false;
      }
    } else if (i == 0 && build_one_tree_pb_file_not_exits_caurse_parse_failed) {
      std::string command = std::string("rm -rf ") + tree_path;
      if (system(command.c_str()) != 0) {
        LOG_ERROR << "execute " << command << " failed";
        return false;
      }
    } else {
      if (!ConvertPbToBinaryFile(tree_path, *(trees[i]))) {
        return false;
      }
    }

    delete trees[i];
  }

  return true;
}

}  // namespace tdm_serving
