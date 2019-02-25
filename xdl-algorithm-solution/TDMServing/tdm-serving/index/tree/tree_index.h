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

#ifndef TDM_SERVING_INDEX_TREE_TREE_INDEX_H_
#define TDM_SERVING_INDEX_TREE_TREE_INDEX_H_

#include <fstream>
#include "index/index.h"
#include "index/tree/tree.h"
#include "index/tree/tree_searcher.h"
#include "index/tree/tree_search_context.h"

namespace tdm_serving {

class TreeIndexConf;

// Tree-based Deep Match Index
class TreeIndex : public Index {
 public:
  TreeIndex();
  virtual ~TreeIndex();

  virtual bool Init(const IndexConf* index_conf);

  virtual bool Search(SearchContext* context,
                      const SearchParam& search_param,
                      SearchResult* search_result);

  virtual SearchContext* GetSearchContext();

  virtual void ReleaseSearchContext(SearchContext* context);

 protected:
  virtual IndexConf* CreateIndexConf();

  virtual TreeSearcher* tree_searcher();

 private:
  bool GenerateResponse(TreeSearchContext* context,
                        const SearchParam& search_param,
                        SearchResult* search_result);

 private:
  const TreeIndexConf* tree_index_conf_;

  // tree data
  Tree tree_;

  // tree searcher
  TreeSearcher tree_searcher_;

  DISALLOW_COPY_AND_ASSIGN(TreeIndex);
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_TREE_TREE_INDEX_H_
