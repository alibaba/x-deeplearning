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

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <set>
#include <thread>

#include "tdm/store.h"
#include "tdm/bitmap.h"

#include "tdm/tree.pb.h"

namespace tdm {

DistTree DistTree::instance_;

DistTree::DistTree(): key_prefix_(), branch_(2), store_(NULL),
                      initialized_(false), max_level_(0) {
}

DistTree::DistTree(const std::string& key_prefix,
                   int branch, Store* store)
    : key_prefix_(key_prefix), branch_(branch), store_(store),
      initialized_(false), max_level_(0) {
  Load();
}

Store* DistTree::store() const {
  return store_;
}

int DistTree::branch() const {
  return branch_;
}

std::string DistTree::key_prefix() const {
  return key_prefix_;
}

void DistTree::set_store(Store* store) {
  store_ = store;
}

void DistTree::set_branch(int branch) {
  branch_ = branch;
}

void DistTree::set_key_prefix(const std::string& key_prefix) {
  key_prefix_ = key_prefix;
  Load();
}

int DistTree::max_level() const {
  return max_level_;
}

bool DistTree::Load() {
  initialized_ = false;
  max_level_ = 0;
  id_code_map_.clear();

  if (store_ == NULL) {
    std::cerr << "Failed to load tree, store is null" << std::endl;
    return false;
  }

  std::string meta_key = key_prefix_;
  meta_key.append(".tree_meta");
  std::string meta_value;
  if (!store_->Get(meta_key, &meta_value)) {
    std::cout << "Get meta info from store failed" << std::endl;
    return false;
  }

  TreeMeta meta;
  if (!meta.ParseFromString(meta_value)) {
    std::cerr << "Parse meta failed" << std::endl;
    return false;
  }

  max_level_ = meta.max_level();
  int64_t max_leaf_id = -1;
  max_code_ = -1;
  for (auto it = meta.id_code_part().begin();
       it != meta.id_code_part().end(); ++it) {
    std::string part_str;
    if (!store_->Get(*it, &part_str)) {
      std::cerr << "Load partition " << *it << " failed!" << std::endl;
      return false;
    }
    IdCodePart part;
    if (!part.ParseFromString(part_str)) {
      std::cerr << "Parse par " << *it << " failed" << std::endl;
      return false;
    }
    for (auto iit = part.id_code_list().begin();
         iit != part.id_code_list().end(); ++iit) {
      id_code_map_.insert(std::make_pair(iit->id(), iit->code()));
      codes_.insert(iit->code());
      if (max_leaf_id < iit->id()) {
        max_leaf_id = iit->id();
      }
      if (max_code_ < iit->code()) {
        max_code_ = iit->code();
      }
    }
  }

  internal_id_start_ = max_leaf_id + 1;
  std::cout << "Load successfully, leaf node count:"
            << id_code_map_.size() << ", internal node id start: "
            << internal_id_start_ << std::endl;
  initialized_ = true;
  return true;
}

void DistTree::Persist(int level) {
  if (!initialized_) {
    return;
  }

  CachedStore* store = dynamic_cast<CachedStore*>(store_);
  if (store == NULL) {  // Unable to cache and persist
    return;
  }

  size_t level_start = 0;
  size_t level_end = branch_ * level_start + 1;
  std::vector<std::string> keys;
  keys.reserve(kBatchSize);
  for (int i = 0; i < level && i < max_level_; ++i) {
    size_t j = level_start;
    while (j < level_end) {
      while (j < level_end && keys.size() < kBatchSize) {
        if (!IsFiltered(j)) {
          keys.push_back(MakeKey(j));
        }
        ++j;
      }
      store->Persist(keys);
      keys.clear();
    }
    level_start = level_end;
    level_end = level_start * branch_ + 1;
  }
}

TreeNode DistTree::Node(const std::string& key) const {
  TreeNode node;
  node.key = key;
  if (initialized_ && KeyExists(key)) {
    if (!store_->Get(key, &node.value)) {
      node.value.clear();
    }
  }
  return node;
}

TreeNode DistTree::Parent(const TreeNode& node) const {
  TreeNode parent;
  if (!initialized_ || !Valid(node)) {
    return parent;
  }

  parent.key = ParentKey(node.key);
  if (!parent.key.empty()) {
    if (!store_->Get(parent.key, &parent.value)) {
      parent.value.clear();
    }
  }

  return parent;
}

std::vector<TreeNode> DistTree::Ancestors(const TreeNode& node) const {
  std::vector<TreeNode> ancestors;
  if (!initialized_ || !Valid(node)) {
    return ancestors;
  }

  std::vector<std::string> ancestor_keys = AncestorKeys(node.key);
  std::vector<std::string> ancestor_values = BatchGet(ancestor_keys);
  ancestors.resize(ancestor_keys.size());
  for (size_t i = 0; i < ancestor_keys.size(); ++i) {
    ancestors[i].key = ancestor_keys[i];
    ancestors[i].value = ancestor_values[i];
  }

  return ancestors;
}

std::vector<TreeNode> DistTree::Children(const TreeNode& node) const {
  std::vector<TreeNode> children;
  if (!initialized_ || !Valid(node)) {
    return children;
  }

  std::vector<std::string> child_keys = ChildKeys(node.key);
  std::vector<std::string> child_values = BatchGet(child_keys);
  for (size_t i = 0; i < child_keys.size(); ++i) {
    TreeNode node = {child_keys[i], child_values[i]};
    if (node.valid()) {
      children.push_back(node);
    }
  }

  return children;
}

std::vector<TreeNode> DistTree::Silbings(const TreeNode& node) const {
  std::vector<TreeNode> silbings;
  if (!initialized_ || !Valid(node)) {
    return silbings;
  }

  std::vector<std::string> silbing_keys = SilbingKeys(node.key);
  std::vector<std::string> silbing_values = BatchGet(silbing_keys);
  for (size_t i = 0; i < silbing_keys.size(); ++i) {
    TreeNode node = {silbing_keys[i], silbing_values[i]};
    if (node.valid()) {
      silbings.push_back(node);
    }
  }

  return silbings;
}

std::vector<TreeNode>
DistTree::RandNeighbors(const TreeNode& node, int k) const {
  std::vector<TreeNode> neighbors;
  if (!initialized_ || !Valid(node)) {
    return neighbors;
  }

  RandNBKeyGen generator(this, k);
  auto neighbor_keys = generator(node.key);
  std::vector<std::string> neighbor_values = BatchGet(neighbor_keys);
  for (size_t i = 0; i < neighbor_keys.size(); ++i) {
    TreeNode node = {neighbor_keys[i], neighbor_values[i]};
    if (node.valid()) {
      neighbors.push_back(node);
    }
  }

  return neighbors;
}

std::vector<TreeNode> DistTree::SelectNeighbors(
    const TreeNode& node, const std::vector<int>& indice) const {
  std::vector<TreeNode> neighbors;
  if (!initialized_ || !Valid(node)) {
    return neighbors;
  }

  std::vector<std::string> neighbor_keys;
  for (auto it = indice.begin(); it != indice.end(); ++it) {
    std::string neighbor_key = NeighborKey(node.key, *it);
    if (!neighbor_key.empty()) {
      neighbor_keys.push_back(neighbor_key);
    }
  }

  std::vector<std::string> neighbor_values = BatchGet(neighbor_keys);
  for (size_t i = 0; i < neighbor_keys.size(); ++i) {
    TreeNode node = {neighbor_keys[i], neighbor_values[i]};
    if (node.valid()) {
      neighbors.push_back(node);
    }
  }

  return neighbors;
}

std::string DistTree::ParentKey(const std::string& key) const {
  size_t key_no = KeyNo(key);
  if (key_no == 0) {
    return "";
  }

  --key_no;
  key_no /= branch_;  // parent key number
  if (IsFiltered(key_no)) {
    return "";
  }

  return MakeKey(key_no);
}

std::vector<std::string> DistTree::ChildKeys(const std::string& key) const {
  std::vector<std::string> child_keys;
  size_t key_no = KeyNo(key);
  for (int i = 0; i < branch_; ++i) {
    size_t child_kno = key_no * branch_ + i + 1;
    if (!IsFiltered(child_kno)) {
      child_keys.push_back(MakeKey(child_kno));
    }
  }

  return child_keys;
}

std::vector<std::string> DistTree::AncestorKeys(const std::string& key) const {
  std::vector<std::string> ancestor_keys;
  size_t key_no = KeyNo(key);
  if (key_no == 0 || IsFiltered(key_no)) {
    return ancestor_keys;
  }

  ancestor_keys.reserve(32);

  while (key_no != 0) {
    ancestor_keys.push_back(MakeKey(key_no));
    --key_no;
    key_no /= branch_;
  }

  return ancestor_keys;
}

std::vector<std::string> DistTree::SilbingKeys(const std::string& key) const {
  std::vector<std::string> silbing_keys;
  size_t key_no = KeyNo(key);
  if (key_no == 0) {
    return silbing_keys;
  }

  silbing_keys.reserve(branch_);
  size_t pk_no = (key_no - 1) / branch_;
  for (int i = 0; i < branch_; ++i) {
    size_t silb_kno = pk_no * branch_ + i + 1;
    if (silb_kno != key_no && !IsFiltered(silb_kno)) {
      silbing_keys.push_back(MakeKey(silb_kno));
    }
  }

  return silbing_keys;
}

std::string DistTree::NeighborKey(const std::string& key, int index) const {
  std::string neighbor_key;
  size_t key_no = KeyNo(key);
  if (key_no == 0) {
    return neighbor_key;
  }

  size_t p = key_no;
  size_t start = 0;
  while (p != 0) {
    --p;
    p /= branch_;
    start = start * branch_ + 1;
  }

  size_t end =  start * branch_ + 1;
  start += index;
  /*
  if (start == key_no) {
    ++start;
  }
  */

  if (start >= end || IsFiltered(start)) {
    return neighbor_key;
  }
  return MakeKey(start);
}

size_t DistTree::KeyNo(const std::string& key) const {
  size_t key_no = 0;
  const unsigned char* ptr = reinterpret_cast<const unsigned char*>(
      key.data() + key_prefix_.size());
  for (size_t i = 0; i < sizeof(size_t); ++i) {
    key_no <<= 8;
    key_no += ptr[i];
  }

  return key_no;
}

int DistTree::NodeLevel(size_t key_no) const {
  int level = 0;
  while (key_no != 0) {
    ++level;
    --key_no;
    key_no /= branch_;
  }

  return level;
}

bool DistTree::KeyExists(const std::string& key) const {
  if (key.size() != key_prefix_.size() + sizeof(size_t)) {
    return false;
  }

  size_t no = KeyNo(key);
  return !IsFiltered(no);
}

std::string DistTree::MakeKey(size_t key_no) const {
  std::string key(key_prefix_);
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

int64_t DistTree::NodeIdToCode(int64_t id) {
  int64_t code = -1;
  if (initialized_) {
    if (id < internal_id_start_) {
      auto it = id_code_map_.find(id);
      if (it != id_code_map_.end()) {
        code = it->second;
      }
    } else {
      code = id - internal_id_start_;
      if (code > max_code_) {
        code = -1;
      }
    }
  }
  return code;
}

TreeNode DistTree::NodeById(int64_t id) {
  TreeNode node;
  int64_t code = NodeIdToCode(id);
  if (code == -1) {
    return node;
  }
  node.key = MakeKey(code);
  if (!store_->Get(node.key, &node.value)) {
    node.value.clear();
  }
  return node;
}

TreeNode DistTree::NodeByCode(int64_t code) {
  TreeNode node;
  if (!initialized_ || IsFiltered(code)) {
    return node;
  }
  node.key = MakeKey(code);
  if (!store_->Get(node.key, &node.value)) {
    node.value.clear();
  }
  return node;
}

///////////////////////// Batch operation /////////////////////////

std::vector<TreeNode>
DistTree::Nodes(const std::vector<std::string>& keys) {
  std::vector<TreeNode> nodes(keys.size());
  if (!initialized_) {
    return nodes;
  }

  std::vector<std::string> valid_keys;
  std::vector<int> index;
  valid_keys.reserve(keys.size());
  index.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    if (KeyExists(keys[i])) {
      valid_keys.push_back(keys[i]);
      index.push_back(i);
    }
  }

  std::vector<std::string> values = BatchGet(valid_keys);
  for (size_t i = 0; i < valid_keys.size(); ++i) {
    TreeNode& node = nodes[index[i]];
    node.key = valid_keys[i];
    node.value = values[i];
  }

  return nodes;
}

std::vector<TreeNode>
DistTree::Parents(const std::vector<TreeNode>& nodes) {
  std::vector<TreeNode> parents(nodes.size());
  if (!initialized_) {
    return parents;
  }

  std::vector<std::string> parent_keys;
  std::vector<int> index;
  parent_keys.reserve(nodes.size());
  index.reserve(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (Valid(nodes[i])) {
      parent_keys.push_back(ParentKey(nodes[i].key));
      index.push_back(i);
    }
  }

  std::vector<std::string> values = BatchGet(parent_keys);
  for (size_t i = 0; i < parent_keys.size(); ++i) {
    TreeNode& parent = parents[index[i]];
    parent.key = parent_keys[i];
    parent.value = values[i];
  }

  return parents;
}

#define GEN_CODE(KEYS_GENERATOR)                                      \
  std::vector<std::vector<TreeNode> > gen_nodes(nodes.size());        \
  if (!initialized_) {                                                \
    return gen_nodes;                                                 \
  }                                                                   \
                                                                      \
  std::vector<std::string> gen_keys;                                  \
  std::vector<int> counts;                                            \
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {            \
    if (Valid(*it)) {                                                 \
      std::vector<std::string> g_keys = KEYS_GENERATOR(it->key);      \
      gen_keys.insert(gen_keys.end(), g_keys.begin(), g_keys.end());  \
      counts.push_back(g_keys.size());                                \
    } else {                                                          \
      counts.push_back(0);                                            \
    }                                                                 \
  }                                                                   \
                                                                      \
  std::vector<std::string> values = BatchGet(gen_keys);               \
  int index = 0;                                                      \
  for (size_t i = 0; i < gen_nodes.size(); ++i) {                     \
    auto& gn = gen_nodes[i];                                          \
    gn.resize(counts[i]);                                             \
    for (int j = 0; j < counts[i]; ++j) {                             \
      TreeNode& node = gn[j];                                         \
      node.key = gen_keys[index + j];                                 \
      node.value = values[index + j];                                 \
    }                                                                 \
    index += counts[i];                                               \
  }                                                                   \
                                                                      \
  return gen_nodes;

std::vector<std::vector<TreeNode> >
DistTree::Ancestors(const std::vector<TreeNode>& nodes) {
  GEN_CODE(AncestorKeys);
}

std::vector<std::vector<TreeNode> >
DistTree::Children(const std::vector<TreeNode>& nodes) {
  GEN_CODE(ChildKeys);
}

std::vector<std::vector<TreeNode> >
DistTree::Silbings(const std::vector<TreeNode>& nodes) {
  GEN_CODE(SilbingKeys);
}

std::vector<std::vector<TreeNode> >
DistTree::RandNeighbors(const std::vector<TreeNode>& nodes, int k) {
  RandNBKeyGen generator(this, k);
  GEN_CODE(generator);
}

std::vector<std::vector<TreeNode> >
DistTree::SelectNeighbors(const std::vector<TreeNode>& nodes,
                          const std::vector<std::vector<int> >& sels) {
  std::unordered_map<std::string, const std::vector<int>* > kmap;
  for (size_t i = 0; i < nodes.size(); ++i) {
    kmap.insert(std::make_pair(nodes[i].key, &sels[i]));
  }
  SelectNBKeyGen generator(this, kmap);
  GEN_CODE(generator);
}

#undef GEN_CODE

std::vector<int64_t> DistTree::NodeIdToCode(const std::vector<int64_t>& ids) {
  std::vector<int64_t> codes(ids.size(), -1);
  if (initialized_) {
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] < internal_id_start_) {
        auto it = id_code_map_.find(ids[i]);
        if (it != id_code_map_.end()) {
          codes[i] = it->second;
        }
      } else {
        codes[i] = ids[i] - internal_id_start_;
        if (codes[i] > max_code_) {
          codes[i] = -1;
        }
      }
    }
  }
  return codes;
}

std::vector<TreeNode> DistTree::NodeById(const std::vector<int64_t>& ids) {
  std::vector<TreeNode> nodes(ids.size());
  if (!initialized_) {
    return nodes;
  }

  std::vector<int64_t> codes = NodeIdToCode(ids);
  std::vector<std::string> keys;
  std::vector<int> index;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (codes[i] != -1) {
      keys.push_back(MakeKey(codes[i]));
      index.push_back(i);
    }
  }

  if (!keys.empty()) {
    std::vector<std::string> values = BatchGet(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      TreeNode& node = nodes[index[i]];
      node.key = keys[i];
      node.value = values[i];
    }
  }

  return nodes;
}

std::vector<TreeNode>
DistTree::NodeByCode(const std::vector<int64_t>& codes) {
  std::vector<TreeNode> nodes(codes.size());
  if (!initialized_) {
    return nodes;
  }

  std::vector<std::string> keys;
  std::vector<int> index;
  for (size_t i = 0; i < codes.size(); ++i) {
    if (!IsFiltered(codes[i])) {
      keys.push_back(MakeKey(codes[i]));
      index.push_back(i);
    }
  }

  if (!keys.empty()) {
    std::vector<std::string> values = BatchGet(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      TreeNode& node = nodes[index[i]];
      node.key = keys[i];
      node.value = values[i];
    }
  }

  return nodes;
}

////////////////////////// Level operation ////////////////////////

DistTree::Iterator::Iterator():
    tree_(NULL), start_no_(0), end_no_(0), cache_start_(0), cache_end_(0) {
}

DistTree::Iterator::~Iterator() {
}

DistTree::Iterator& DistTree::Iterator::Next() {
  if (tree_ != NULL && end_no_ > start_no_) {
    size_t key_no = tree_->KeyNo(node_.key);
    ++key_no;
    while (key_no < end_no_ && tree_->IsFiltered(key_no)) {
      ++key_no;
    }

    if (key_no < end_no_) {
      CachedStore* store = dynamic_cast<CachedStore*>(tree_->store_);
      if (store != NULL && (key_no < cache_start_ || key_no > cache_end_)) {
        AdvanceCache(key_no, kBatchSize);
      }
      node_ = tree_->Node(tree_->MakeKey(key_no));
    } else {
      node_.key = tree_->MakeKey(key_no);
    }
  }

  return *this;
}

DistTree::Iterator& DistTree::Iterator::Back() {
  if (tree_ != NULL && end_no_ > start_no_) {
    size_t key_no = tree_->KeyNo(node_.key);
    --key_no;
    while (key_no >= start_no_ && tree_->IsFiltered(key_no)) {
      --key_no;
    }

    if (key_no >= start_no_) {
      CachedStore* store = dynamic_cast<CachedStore*>(tree_->store_);
      if (store != NULL && (key_no < cache_start_ || key_no > cache_end_)) {
        BackwardCache(key_no, kBatchSize);
      }
      node_ = tree_->Node(tree_->MakeKey(key_no));
    }
  }

  return *this;
}

void DistTree::Iterator::AdvanceCache(size_t key_no, int number) {
  CachedStore* store = dynamic_cast<CachedStore*>(tree_->store_);
  if (!tree_->initialized_ || !store->enable_cache()) {
    return;
  }

  std::vector<std::string> keys;
  keys.reserve(number);

  cache_start_ = key_no;
  cache_end_ = key_no;
  keys.push_back(tree_->MakeKey(key_no++));
  while (key_no < end_no_ && keys.size() < number) {
    if (!tree_->IsFiltered(key_no)) {
      keys.push_back(tree_->MakeKey(key_no));
    }
    ++key_no;
    ++cache_end_;
  }

  // store.Cache(keys);
  tree_->BatchGet(keys);
}

void DistTree::Iterator::BackwardCache(size_t key_no, int number) {
  CachedStore* store = dynamic_cast<CachedStore*>(tree_->store_);
  if (!tree_->initialized_ || !store->enable_cache()) {
    return;
  }

  std::vector<std::string> keys;
  keys.reserve(number);

  cache_start_ = key_no;
  cache_end_ = key_no;
  keys.push_back(tree_->MakeKey(key_no++));
  while (key_no >= start_no_ && keys.size() < number) {
    if (!tree_->IsFiltered(key_no)) {
      keys.push_back(tree_->MakeKey(key_no));
    }
    --key_no;
    --cache_start_;
  }

  tree_->BatchGet(keys);
}

TreeNode& DistTree::Iterator::operator*() {
  return node_;
}

TreeNode* DistTree::Iterator::operator->() {
  return &node_;
}

DistTree::Iterator& DistTree::Iterator::operator++() {
  return Next();
}

DistTree::Iterator DistTree::Iterator::operator++(int) {
  Iterator copy = *this;
  Next();
  return copy;
}

DistTree::Iterator& DistTree::Iterator::operator--() {
  return Back();
}

DistTree::Iterator DistTree::Iterator::operator--(int) {
  Iterator copy = *this;
  Back();
  return copy;
}

bool DistTree::Iterator::operator<(const Iterator& other) {
  return this->node_.key < other.node_.key;
}

bool DistTree::Iterator::operator==(const Iterator& other) {
  return this->node_.key == other.node_.key;
}

DistTree::Iterator DistTree::LevelIterator(int level) {
  Iterator it;
  if (level >= max_level_) {
    return it;
  }

  it.tree_ = this;
  size_t start = 0;
  while (level > 0) {
    start = start * branch_ + 1;
    --level;
  }
  it.start_no_ = start;
  it.end_no_ = start * branch_ + 1;
  while (start < it.end_no_ && IsFiltered(start)) {
    ++start;
  }

  it.node_ = Node(MakeKey(start));
  return it;
}

DistTree::Iterator DistTree::LevelEnd(int level) {
  Iterator it;
  if (level >= max_level_) {
    return it;
  }

  size_t start = 0;
  while (level > 0) {
    start = start * branch_ + 1;
    --level;
  }
  it.start_no_ = start;
  it.end_no_ = start * branch_ + 1;

  it.node_.key = MakeKey(it.end_no_);
  return it;
}


///////////////////// Helper Function ////////////////////////

std::vector<std::string>
DistTree::BatchGet(const std::vector<std::string>& keys) const {
  std::vector<std::string> values;
  std::vector<std::string> round_keys;
  for (auto it = keys.begin(); it != keys.end(); ++it) {
    round_keys.push_back(*it);
    if (round_keys.size() >= kBatchSize) {
      std::vector<std::string> round_values(round_keys.size());
      store_->MGet(round_keys, &round_values);
      values.insert(values.end(), round_values.begin(), round_values.end());
      round_keys.clear();
    }
  }
  if (!round_keys.empty()) {
    std::vector<std::string> round_values(round_keys.size());
    store_->MGet(round_keys, &round_values);
    values.insert(values.end(), round_values.begin(), round_values.end());
  }
  return values;
}

DistTree::RandNBKeyGen::RandNBKeyGen(const DistTree* tree, int k)
    : tree_(tree), count_(k) {
}

std::vector<std::string>
DistTree::RandNBKeyGen::operator()(const std::string& key) {
  std::vector<std::string> neighbor_keys;
  size_t key_no = tree_->KeyNo(key);
  size_t origin_key_no = key_no;
  if (key_no == 0) {  // root node
    return neighbor_keys;
  }

  int level = 0;
  while (key_no != 0) {
    ++level;
    --key_no;
    key_no /= tree_->branch_;
  }

  int k = count_;
  size_t neighbor_count = static_cast<size_t>(pow(tree_->branch_, level)) - 1;
  if (static_cast<size_t>(k) > neighbor_count) {
    k = neighbor_count;
  }

  // 随机数种子
  static __thread std::hash<std::thread::id> hasher;
  static __thread std::mt19937
      rng(clock() + hasher(std::this_thread::get_id()));
  std::uniform_int_distribution<int> distribution(0, neighbor_count - 1);
  while (neighbor_keys.size() < static_cast<size_t>(k)) {
    int i = distribution(rng);
    std::string neighbor_key = tree_->NeighborKey(key, i);
    if (!neighbor_key.empty()) {
      neighbor_keys.push_back(neighbor_key);
    }
  }

  return neighbor_keys;
}

DistTree::SelectNBKeyGen::SelectNBKeyGen(
    const DistTree* tree, const KSelMap& kmap): tree_(tree), kmap_(kmap) {
}

std::vector<std::string>
DistTree::SelectNBKeyGen::operator()(const std::string& key) {
  std::vector<std::string> neighbor_keys;
  auto it = kmap_.find(key);
  if (it == kmap_.end()) {
    return neighbor_keys;
  }

  auto& indice = *it->second;
  for (auto it = indice.begin(); it != indice.end(); ++it) {
    std::string neighbor_key = tree_->NeighborKey(key, *it);
    if (!neighbor_key.empty()) {
      neighbor_keys.push_back(neighbor_key);
    }
  }
  return neighbor_keys;
}

}  // namespace tdm
