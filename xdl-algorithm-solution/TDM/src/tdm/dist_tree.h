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

#ifndef TDM_DIST_TREE_H_
#define TDM_DIST_TREE_H_

#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "tdm/store.h"

namespace tdm {

// Key format: prefix + [8 bytes index]
struct TreeNode {
  std::string key;
  std::string value;

  bool valid() const {
    return (!key.empty()) && (!value.empty());
  }
};

class DistTree {
 public:
  DistTree();
  DistTree(const std::string& key_prefix,
           int branch, Store* store);

  bool Load();
  void Persist(int level);  // Persist [0, level) top layer

  Store* store() const;
  int branch() const;
  std::string key_prefix() const;
  int max_level() const;

  void set_store(Store* store);
  void set_branch(int branch);
  void set_key_prefix(const std::string& key_prefix);

  static DistTree& GetInstance() {
    return instance_;
  }

  TreeNode Node(const std::string& key) const;
  TreeNode Parent(const TreeNode& node) const;
  std::vector<TreeNode> Ancestors(const TreeNode& node) const;
  std::vector<TreeNode> Children(const TreeNode& node) const;
  std::vector<TreeNode> Silbings(const TreeNode& node) const;
  std::vector<TreeNode> RandNeighbors(const TreeNode& node, int k) const;
  std::vector<TreeNode> SelectNeighbors(const TreeNode& node,
                                        const std::vector<int>& indice) const;

  size_t KeyNo(const std::string& key) const;
  int NodeLevel(size_t key_no) const;
  std::string MakeKey(size_t key_no) const;

  bool Valid(const TreeNode& node) const {
    return node.valid() && KeyExists(node.key);
  }

  int64_t NodeIdToCode(int64_t id);
  TreeNode NodeById(int64_t id);
  TreeNode NodeByCode(int64_t code);

  ////////////////// Batch Operation ////////////////////

  std::vector<TreeNode> Nodes(const std::vector<std::string>& keys);
  std::vector<TreeNode> Parents(const std::vector<TreeNode>& nodes);

  std::vector<std::vector<TreeNode> >
  Ancestors(const std::vector<TreeNode>& nodes);

  std::vector<std::vector<TreeNode> >
  Children(const std::vector<TreeNode>& nodes);

  std::vector<std::vector<TreeNode> >
  Silbings(const std::vector<TreeNode>& nodes);

  std::vector<int64_t> NodeIdToCode(const std::vector<int64_t>& ids);

  std::vector<TreeNode> NodeById(const std::vector<int64_t>& ids);
  std::vector<TreeNode> NodeByCode(const std::vector<int64_t>& ids);

  std::vector<std::vector<TreeNode> >
  RandNeighbors(const std::vector<TreeNode>& nodes, int k);

  std::vector<std::vector<TreeNode> >
  SelectNeighbors(const std::vector<TreeNode>& nodes,
                  const std::vector<std::vector<int> >& sels);

  inline bool IsFiltered(size_t key_no) const {
    size_t max_key_no = 0;
    for (int i = 0; i < max_level_; ++i) {
      max_key_no = max_key_no * branch_ + 1;
    }

    while (key_no < max_key_no) {
      if (codes_.find(key_no) != codes_.end()) {
        return false;
      }
      key_no = key_no * branch_ + 1;  // Left deep
    }

    return true;
  }

  ////////////////// Level Operation /////////////////////////

  class Iterator {
    friend class DistTree;
   public:
    Iterator();
    ~Iterator();

    Iterator& Next();
    Iterator& Back();

    TreeNode& operator*();
    TreeNode* operator->();
    Iterator& operator++();
    Iterator operator++(int);
    Iterator& operator--();
    Iterator operator--(int);
    bool operator<(const Iterator& other);
    bool operator==(const Iterator& other);
    bool operator!=(const Iterator& other) {
      return !this->operator==(other);
    }

   private:
    void AdvanceCache(size_t key_no, int number);
    void BackwardCache(size_t key_no, int number);

   private:
    DistTree* tree_;
    TreeNode node_;
    size_t start_no_;
    size_t end_no_;
    size_t cache_start_;
    size_t cache_end_;
  };

  Iterator LevelIterator(int level);
  Iterator LevelEnd(int level);

 private:
  class RandNBKeyGen {
   public:
    explicit RandNBKeyGen(const DistTree* tree, int k);
    std::vector<std::string> operator()(const std::string& key);

   private:
    const DistTree* tree_;
    int count_;
  };

  using KSelMap = std::unordered_map<std::string, const std::vector<int>* >;
  class SelectNBKeyGen {
   public:
    explicit SelectNBKeyGen(const DistTree* tree, const KSelMap& kmap);
    std::vector<std::string> operator()(const std::string& key);

   private:
    const DistTree* tree_;
    const KSelMap& kmap_;
  };

 private:
  std::string ParentKey(const std::string& key) const;
  std::vector<std::string> ChildKeys(const std::string& key) const;
  std::vector<std::string> AncestorKeys(const std::string& key) const;
  std::vector<std::string> SilbingKeys(const std::string& key) const;
  std::string NeighborKey(const std::string& key, int index) const;
  bool KeyExists(const std::string& key) const;

  std::vector<std::string> BatchGet(const std::vector<std::string>& keys) const;

 private:
  std::string key_prefix_;
  int branch_;
  Store* store_;
  bool initialized_;
  int max_level_;
  std::unordered_map<int64_t, int64_t> id_code_map_;
  std::unordered_set<int64_t> codes_;
  int64_t internal_id_start_;
  int64_t max_code_;
  static DistTree instance_;
};

}  // namespace tdm

#endif  // TDM_DIST_TREE_H_
