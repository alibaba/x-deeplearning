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

// Copyright 2018 Alibaba Inc. All Rights Reserved.

#include "tdm/dist_tree.h"

#include "gtest/gtest.h"

#include "tdm/local_store.h"
#include "tdm/tree.pb.h"

namespace tdm {

class MockStore: public Store {
 public:
  bool Get(const std::string& key, std::string* value) override {
    if (key == "root.tree_meta") {
      TreeMeta meta;
      meta.set_max_level(20);
      auto part_id = meta.mutable_id_code_part()->Add();
      part_id->assign("part");
      meta.SerializeToString(value);
    } else if (key == "part") {
      IdCodePart part;
      part.set_part_id("part");
      for (size_t i = (1 << 10); i < (1 << 11); ++i) {
        auto id_code = part.mutable_id_code_list()->Add();
        id_code->set_id(i);
        id_code->set_code(i);
      }
      part.SerializeToString(value);
    } else {
      value->assign(key);
    }

    return true;
  }

  bool Put(const std::string& key, const std::string& value) override {
    (void) key;
    (void) value;
    return true;
  }

  std::vector<bool> MGet(const std::vector<std::string>& keys,
                         std::vector<std::string>* values) override {
    std::vector<bool> ret(keys.size(), true);
    for (size_t i = 0; i < keys.size(); ++i) {
      (*values)[i].assign(keys[i]);
    }
    return ret;
  }

  std::vector<bool> MPut(const std::vector<std::string>& keys,
                         const std::vector<std::string>& values) override {
    (void) values;
    std::vector<bool> ret(keys.size(), true);
    return ret;
  }

  bool Remove(const std::string& key) {
    return true;
  }

  bool Dump(const std::string& filename) {
	(void) filename;
	return true;
  }
};

std::string MakeKey(const std::string& prefix, size_t key_no) {
  std::string key(prefix);
  unsigned char buffer[sizeof(size_t)];
  memset(buffer, 0x00, sizeof(buffer));
  unsigned char* ptr = buffer + sizeof(size_t) - 1;

  while (key_no != 0) {
    *ptr = key_no & 0xFF;
    key_no >>= 8;
    --ptr;
  }
  key.append(reinterpret_cast<char*>(buffer), sizeof(size_t));
  return key;
}

TEST(DistTree, TestNode) {
  MockStore store;
  DistTree tree("root", 2, &store);
  std::string key = MakeKey("root", 0);
  TreeNode root = tree.Node(key);
  ASSERT_TRUE(root.valid());
}

TEST(DistTree, TestParent) {
  MockStore store;
  DistTree tree("root", 2, &store);
  std::string key = MakeKey("root", 0);
  TreeNode root = tree.Node(key);
  ASSERT_TRUE(root.valid());

  TreeNode parent = tree.Parent(root);
  ASSERT_FALSE(parent.valid());

  std::vector<TreeNode> children = tree.Children(root);
  ASSERT_EQ(2ul, children.size());
  ASSERT_TRUE(children[0].valid() &&  children[1].valid());
  std::string key0 = MakeKey("root", 1);
  ASSERT_EQ(children[0].key, key0);
  std::string key1 = MakeKey("root", 2);
  ASSERT_EQ(children[1].key, key1);

  parent = tree.Parent(children[0]);
  ASSERT_EQ(parent.key, root.key);
}

TEST(DistTree, TestChildren) {
  MockStore store;
  DistTree tree("root", 4, &store);

  std::string key = MakeKey("root", 0);
  TreeNode root = tree.Node(key);
  std::vector<TreeNode> children =  tree.Children(root);
  ASSERT_EQ(2ul, children.size());

  key = MakeKey("root", 1);
  ASSERT_EQ(key, children[0].key);

  key = MakeKey("root", 4);
  ASSERT_EQ(key, children[1].key);
}

TEST(DistTree, TestAncestors) {
  MockStore store;
  DistTree tree("root", 4, &store);

  TreeNode node = tree.Node(MakeKey("root", 500));
  ASSERT_TRUE(node.valid());

  std::vector<TreeNode> ancestors = tree.Ancestors(node);
  ASSERT_EQ(5, ancestors.size());
  ASSERT_EQ(MakeKey("root", 124), ancestors[1].key);
}

TEST(DistTree, TestSilbings) {
  MockStore store;
  DistTree tree("root", 4, &store);

  TreeNode node = tree.Node(MakeKey("root", 121));
  ASSERT_TRUE(node.valid());

  std::vector<TreeNode> silbings = tree.Silbings(node);
  ASSERT_EQ(3ul, silbings.size());
  ASSERT_EQ(MakeKey("root", 122), silbings[0].key);
  ASSERT_EQ(MakeKey("root", 123), silbings[1].key);
  ASSERT_EQ(MakeKey("root", 124), silbings[2].key);
}

TEST(DistTree, TestRandNeighbors) {
  MockStore store;
  DistTree tree("root", 4, &store);

  TreeNode node = tree.Node(MakeKey("root", 121));
  ASSERT_TRUE(node.valid());

  std::vector<TreeNode> neighbors = tree.RandNeighbors(node, 5);
  ASSERT_EQ(5ul, neighbors.size());

  for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
    std::cout << it->key << std::endl;
  }
}


TEST(DistTree, TestSelectNeighbors) {
  MockStore store;
  DistTree tree("root", 4, &store);

  TreeNode node = tree.Node(MakeKey("root", 121));
  ASSERT_TRUE(node.valid());

  std::vector<int> indice = {1, 3, 5, 7, 9};
  std::vector<TreeNode> neighbors = tree.SelectNeighbors(node, indice);
  ASSERT_EQ(5ul, neighbors.size());

  ASSERT_EQ(MakeKey("root", 86), neighbors[0].key);
  ASSERT_EQ(MakeKey("root", 88), neighbors[1].key);
  ASSERT_EQ(MakeKey("root", 90), neighbors[2].key);
  ASSERT_EQ(MakeKey("root", 92), neighbors[3].key);
}

TEST(DistTree, TestBuild) {
  LocalStore store;
  ASSERT_TRUE(store.Init(""));
  DistTree tree("root", 2, &store);

  size_t level = 20;
  size_t max_id = (1 << level) - 1;
  size_t first_leaf_id = (1 << (level - 1)) - 1;
  std::cout << "Max id: " << max_id << ", First leaf id: "
            << first_leaf_id << std::endl;
  Node node;
  TreeMeta meta;
  std::vector<IdCodePart*> id_code_part;
  meta.set_max_level(level);
  std::vector<std::string> keys;
  std::vector<std::string> values;
  for (size_t i = 0; i < max_id; ++i) {
    std::string code = tree.MakeKey(i);
    std::string value;
    node.set_id(i);
    node.set_probality(0.0);
    node.set_leaf_cate_id(0);
    node.set_is_leaf(i >= first_leaf_id);
    ASSERT_TRUE(node.SerializeToString(&value));
    keys.push_back(code);
    values.push_back(value);
    if (keys.size() >= 128) {
      auto vec = store.MPut(keys, values);
      for (auto it = vec.begin(); it != vec.end(); ++it) {
        ASSERT_TRUE(*it);
      }
      keys.clear();
      values.clear();
    }
    // ASSERT_TRUE(store.Put(code, value));
    IdCodePart* part = NULL;
    if (id_code_part.empty() ||
        id_code_part.back()->id_code_list().size() == 512) {
      auto part = new IdCodePart();
      part->set_part_id(MakeKey("Part_", id_code_part.size() + 1));
      id_code_part.push_back(part);
    }

    part = id_code_part.back();
    auto id_code = part->mutable_id_code_list()->Add();
    id_code->set_id(i);
    id_code->set_code(i);
  }

  if (!keys.empty()) {
    auto vec = store.MPut(keys, values);
    for (auto it = vec.begin(); it != vec.end(); ++it) {
      ASSERT_TRUE(*it);
    }
    keys.clear();
    values.clear();
  }

  for (size_t i = 0; i < id_code_part.size(); ++i) {
    auto part_id = meta.mutable_id_code_part()->Add();
    auto part = id_code_part[i];
    part_id->assign(part->part_id());
    std::string value;
    ASSERT_TRUE(part->SerializeToString(&value));
    ASSERT_TRUE(store.Put(part->part_id(), value));
    delete part;
  }

  std::string meta_str;
  ASSERT_TRUE(meta.SerializeToString(&meta_str));
  ASSERT_TRUE(store.Put("root.tree_meta", meta_str));
  ASSERT_TRUE(store.Dump("local_store.pb"));
}

TEST(DistTree, TestLoad) {
  LocalStore store;
  ASSERT_TRUE(store.Init(""));
  store.LoadData("local_store.pb");
  DistTree tree("root", 2, &store);
  TreeNode root = tree.Node(tree.MakeKey(0));
  ASSERT_TRUE(tree.Valid(root));
}

TEST(DistTree, TesLevelTraverse) {
  LocalStore store;
  ASSERT_TRUE(store.Init(""));
  store.LoadData("local_store.pb");
  DistTree tree("root", 2, &store);
  for (int i = 0; i < tree.max_level(); ++i) {
    auto it = tree.LevelIterator(i);
    size_t level_key_no = tree.KeyNo(it->key);
    auto end = tree.LevelEnd(i);
    while (it != end) {
      auto& node = *it;
      ASSERT_EQ(level_key_no, tree.KeyNo(node.key));
      it.Next();
      ++level_key_no;
    }
  }
}

}  // namespace tdm
