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

#ifndef TDM_SERVING_INDEX_TREE_TREE_SEARCHER_H_
#define TDM_SERVING_INDEX_TREE_TREE_SEARCHER_H_

#include <vector>
#include "common/common_def.h"
#include "proto/search.pb.h"

namespace tdm_serving {

class TreeIndexConf;
class Tree;
class TreeSearchContext;
class NodeScore;
class ItemFeature;

class TreeSearcher {
 public:
  TreeSearcher();

  virtual ~TreeSearcher();

  bool Init(const TreeIndexConf* index_conf);

  bool Search(Tree* tree,
              TreeSearchContext* context,
              const SearchParam& search_param);

 protected:
  // Calculate node scores by accessing model layer
  virtual bool CalculateScore(TreeSearchContext* context,
                              const SearchParam& search_param,
                              std::vector<ItemFeature*>* item_features,
                              std::vector<NodeScore*>* node_scores);

 private:
  // Calculate score for each candidate node
  bool CalculateNodes(TreeSearchContext* context,
                      const SearchParam& search_param,
                      uint32_t level, uint32_t max_level);

  // Sort nodes by score
  void SortNodes(TreeSearchContext* context,
                 const SearchParam& search_param,
                 uint32_t level, uint32_t max_level);

  // Get winner nodes and set their children to be new candidates
  void SpreadNodes(TreeSearchContext* context,
                   const SearchParam& search_param,
                   uint32_t level);

 private:
  const TreeIndexConf* index_conf_;

  DISALLOW_COPY_AND_ASSIGN(TreeSearcher);
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_TREE_TREE_SEARCHER_H_
