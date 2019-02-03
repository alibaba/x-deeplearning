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

#include "index/tree/tree.h"
#include <omp.h>
#include "index/tree/tree_def.h"
#include "index/tree/tree_index_conf.h"
#include "util/str_util.h"

namespace tdm_serving {

static const uint32_t kMaxBufferSize = 1024;

Tree::Tree()
    : nodes_(NULL),
    node_maps_(NULL) {
}

Tree::~Tree() {
  DELETE_ARRAY(nodes_);

  for (uint32_t i = 0; i < pb_trees_.size(); ++i) {
    DELETE_AND_SET_NULL(pb_trees_[i]);
  }
  pb_trees_.clear();

  DELETE_AND_SET_NULL(node_maps_);
}

bool ComparePbMeta(const UIHead* const & h1, const UIHead* const & h2) {
  return h1->offset() < h2->offset();
}

bool Tree::Init(const TreeIndexConf* index_conf) {
  index_conf_ = index_conf;

  std::string section = index_conf_->section();

  LOG_INFO << "[" << section << "] begin init tree";

  // load meta
  LOG_INFO << "[" << section << "] begin load tree meta";
  std::string meta_file_path =
      index_conf_->latest_index_path() + "/" + kTreeMetaFileName;
  if (!ParsePbFromBinaryFile(meta_file_path, &pb_meta_)) {
    LOG_ERROR << "ParsePbFromBinaryFile for meta file: "
                   << meta_file_path << " failed";
    return false;
  }

  if (pb_meta_.heads_size() == 0) {
    LOG_ERROR << "pb meta heads size is 0";
    return false;
  }

  // sort by offset
  std::sort(pb_meta_.mutable_heads()->mutable_data(),
            pb_meta_.mutable_heads()->mutable_data() + pb_meta_.heads_size(),
            ComparePbMeta);

  // check meta
  tree_meta_.total_node_num_ = 0;
  for (int32_t i = 0; i < pb_meta_.heads_size(); ++i) {
    const UIHead& cur_head = pb_meta_.heads(i);
    if (static_cast<uint32_t>(cur_head.offset()) !=
        tree_meta_.total_node_num_) {
      LOG_ERROR << "cur head offset: " << cur_head.offset()
                     << " not equal total_node_num: "
                     << tree_meta_.total_node_num_;
      return false;
    }
    tree_meta_.total_node_num_ += cur_head.count();
  }

  LOG_INFO << "[" << section << "] tree meta load success";

  // load tree data
  LOG_INFO << "[" << section << "] begin load tree data";
  pb_trees_.resize(pb_meta_.heads_size());
  bool load_pb_tree_st = true;
#pragma omp parallel for num_threads(index_conf_->build_omp())
  for (int32_t i = 0; i < pb_meta_.heads_size(); ++i) {
    const UIHead& cur_head = pb_meta_.heads(i);

    UITree* pb_tree = new UITree();
    char buffer[kMaxBufferSize];
    snprintf(buffer, kMaxBufferSize, "%s/%s%u",
        index_conf_->latest_index_path().c_str(),
        kTreeDataFilePrefix.c_str(), cur_head.tid());
    LOG_INFO << "[" << section << "] load tree data, idx: " << i <<
        ", path: " << buffer;
    if (!ParsePbFromBinaryFile(std::string(buffer), pb_tree)) {
      LOG_ERROR << "ParsePbFromBinaryFile from " << buffer << " failed";
      load_pb_tree_st = false;
      delete pb_tree;
    } else {
      pb_trees_[i] = pb_tree;
    }

    if (pb_tree->nodes_size() != cur_head.count()) {
      LOG_ERROR << "tree node size: " << pb_tree->nodes_size()
                     << " not equal meta head node size: " << cur_head.count();
      load_pb_tree_st = false;
    }
  }
  if (!load_pb_tree_st) {
    LOG_ERROR << "load pb tree failed";
    return false;
  }

  LOG_INFO << "[" << section << "] "
                << "tree data load success, size: " << pb_trees_.size();

  // rebuild tree
  LOG_INFO << "[" << section << "] begin rebuild tree";
  nodes_ = new Node[tree_meta_.total_node_num_];
  bool rebuild_st = true;
#pragma omp parallel for num_threads(index_conf_->build_omp())
  for (size_t i = 0; i < pb_trees_.size(); ++i) {
    LOG_INFO << "[" << section << "] rebuild tree idx: " << i;
    UITree* pb_tree = pb_trees_[i];
    int32_t pos = pb_meta_.heads(i).offset();
    for (int32_t j = 0; j < pb_tree->nodes_size(); ++j) {
      UINode* node_info = pb_tree->mutable_nodes(j);
      if (pos + j != node_info->seq()) {
        LOG_ERROR << "node seq id: " << node_info->seq()
                       << " not equal expected: " << pos + j;
        rebuild_st = false;
      }
      Node* node = nodes_ + node_info->seq();
      node->set_node_info(node_info);
      if (!node->InitNodeStructure(this)) {
        LOG_ERROR << "InitNodeStructure for node seq: " << i << " failed";
        rebuild_st = false;
      }
    }
  }
  if (!rebuild_st) {
    LOG_ERROR << "rebuild st failed";
    return false;
  }

  LOG_INFO << "[" << section << "] tree rebuild success";

  // build node maps
  LOG_INFO << "[" << section << "] begin build node maps";
  node_maps_ = new NodeMaps(index_conf_->build_omp());
  bool build_node_maps_st = true;
#pragma omp parallel num_threads(index_conf_->build_omp())
  {
    uint32_t thread_id = omp_get_thread_num();
    uint32_t total_threads = omp_get_num_threads();
    if (total_threads != index_conf_->build_omp()) {
      LOG_ERROR << "total threads num: " << total_threads
                     << " not equal expected build threads num: "
                     << index_conf_->build_omp();
      build_node_maps_st = false;
    } {
      uint32_t max_level = 0;
      LOG_INFO <<
          "[" << section << "] rebuild node maps, thread idx: " << thread_id;
      for (uint32_t i = 0; i < tree_meta_.total_node_num_; ++i) {
        Node* node = nodes_ + i;
        uint64_t node_id = node->node_info()->id();
        uint32_t pos_id = node_maps_->GetSplitPos(node_id);
        if (pos_id == thread_id) {
          if (!node_maps_->InsertNode(node_id, pos_id, node)) {
            build_node_maps_st = false;
          }
          // set max level
          if (static_cast<uint32_t>(node->node_info()->level()) >
              max_level) {
            max_level = node->node_info()->level();
          }

          std::string node_id = util::ToString(node->node_info()->id());
          uint64_t node_id_hash = node->node_info()->id();
          node->node_info()->set_hashid(node_id_hash);
        }
      }
      util::SimpleMutex::Locker slock(&mutex_);
      if (max_level > tree_meta_.max_level_) {
        tree_meta_.max_level_ = max_level;
      }
    }
  }

  if (!build_node_maps_st) {
    LOG_ERROR << "build node maps failed";
    return false;
  }

  LOG_INFO << "[" << section << "] load tree index successfully";
  LOG_INFO << "---> tree max level: " << tree_meta_.max_level_;
  LOG_INFO << "---> tree total node num: " << tree_meta_.total_node_num_;

  return true;
}

}  // namespace tdm_serving
