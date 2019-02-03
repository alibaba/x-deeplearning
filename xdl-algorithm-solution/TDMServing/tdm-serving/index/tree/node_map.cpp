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

#include <omp.h>
#include "index/tree/node_map.h"
#include "util/log.h"

namespace tdm_serving {

NodeMaps::NodeMaps(uint32_t split_num)
    : split_num_(split_num) {
  for (uint32_t i = 0; i < split_num_; ++i) {
    NodeMap* node_map = new NodeMap();
    node_map_array_.push_back(node_map);
  }
}

NodeMaps::~NodeMaps() {
  for (uint32_t i = 0; i < split_num_; ++i) {
    DELETE_AND_SET_NULL(node_map_array_[i]);
  }
}

bool NodeMaps::InsertNode(uint64_t node_id, Node* node) {
  return InsertNode(node_id, GetSplitPos(node_id), node);
}

bool NodeMaps::InsertNode(uint64_t node_id, uint32_t split_pos, Node* node) {
  NodeMap* node_map = node_map_array_[split_pos];
  NodeMap::iterator iter = node_map->find(node_id);
  if (iter != node_map->end()) {
    LOG_WARN <<
        "duplidate node_id: " << node_id << ", with split_pos: " << split_pos;
    return false;
  }
  (*node_map)[node_id] = node;
  return true;
}

Node* NodeMaps::GetNode(uint64_t node_id) {
  uint32_t split_pos = GetSplitPos(node_id);
  NodeMap* node_map = node_map_array_[split_pos];
  NodeMap::iterator iter = node_map->find(node_id);
  if (iter == node_map->end()) {
    return NULL;
  }
  return iter->second;
}

uint32_t NodeMaps::GetSplitPos(uint64_t node_id) {
  return node_id % split_num_;
}

}  // namespace tdm_serving
