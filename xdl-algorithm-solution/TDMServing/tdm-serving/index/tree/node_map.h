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

#ifndef TDM_SERVING_INDEX_TREE_NODE_MAP_H_
#define TDM_SERVING_INDEX_TREE_NODE_MAP_H_

#include <unordered_map>
#include "common/common_def.h"
#include "index/tree/node.h"

namespace tdm_serving {

typedef std::unordered_map<uint64_t, Node*> NodeMap;

class NodeMaps {
 public:
  explicit NodeMaps(uint32_t split_num);
  ~NodeMaps();

  bool InsertNode(uint64_t node_id, Node* node);
  bool InsertNode(uint64_t node_id, uint32_t split_pos, Node* node);
  Node* GetNode(uint64_t node_id);
  uint32_t GetSplitPos(uint64_t node_id);

 private:
  uint32_t split_num_;
  std::vector<NodeMap*> node_map_array_;

  DISALLOW_COPY_AND_ASSIGN(NodeMaps);
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_TREE_NODE_MAP_H_
