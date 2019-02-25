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

#ifndef TDM_SERVING_INDEX_TREE_TREE_META_H_
#define TDM_SERVING_INDEX_TREE_TREE_META_H_

#include "common/common_def.h"
#include "index/tree/node.h"

namespace tdm_serving {

struct TreeMeta {
  uint32_t total_node_num_;
  // level starts with 0
  uint32_t max_level_;

  TreeMeta()
      : total_node_num_(0),
      max_level_(0) {
  }
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_TREE_TREE_META_H_
