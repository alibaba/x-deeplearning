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

#include "index/tree/tree_index.h"
#include <algorithm>
#include "index/tree/tree_def.h"
#include "index/tree/tree_index_conf.h"
#include "util/str_util.h"
#include "util/object_free_list.h"
#include "util/log.h"

namespace tdm_serving {

TreeIndex::TreeIndex() {
}

TreeIndex::~TreeIndex() {
}

bool TreeIndex::Init(const IndexConf* index_conf) {
  tree_index_conf_ = static_cast<const TreeIndexConf*>(index_conf);

  const std::string& section = tree_index_conf_->section();

  if (!Index::Init(index_conf)) {
    LOG_ERROR << "[" << section << "] Index::Init failed";
    return false;
  }

  // init tree
  if (!tree_.Init(tree_index_conf_)) {
    LOG_ERROR << "[" << section << "] init tree failed";
    return false;
  }

  // init tree searcher
  if (!tree_searcher()->Init(tree_index_conf_)) {
    LOG_ERROR << "[" << section << "] init tree searcher failed";
    return false;
  }

  return true;
}

bool TreeIndex::Search(SearchContext* context,
                       const SearchParam& search_param,
                       SearchResult* search_result) {
  if (search_result == NULL) {
    LOG_WARN << "[" << tree_index_conf_->section() << "] "
                  << "Index Search find illegal parameters";
    return false;
  }

  TreeSearchContext* tree_ctx = static_cast<TreeSearchContext*>(context);

  // do search
  if (!tree_searcher()->Search(&tree_, tree_ctx, search_param)) {
    LOG_WARN << "[" << tree_index_conf_->section() << "] "
                  << "Tree search failed";
    return false;
  }

  // generate response
  if (!GenerateResponse(tree_ctx, search_param, search_result)) {
    LOG_ERROR << "[" << tree_index_conf_->section() << "] "
                   << "Generate response failed";
    return false;
  }

  return true;
}

SearchContext* TreeIndex::GetSearchContext() {
  return util::ObjList<TreeSearchContext>::Instance().Get();
}

void TreeIndex::ReleaseSearchContext(SearchContext* context) {
  util::ObjList<TreeSearchContext>::Instance().Free(
      static_cast<TreeSearchContext*>(context));
}

IndexConf* TreeIndex::CreateIndexConf() {
  return new TreeIndexConf();
}

TreeSearcher* TreeIndex::tree_searcher() {
  return &tree_searcher_;
}

bool TreeIndex::GenerateResponse(
    TreeSearchContext* context,
    const SearchParam& search_param,
    SearchResult* search_result) {
  uint32_t search_topn = search_param.topn();
  NodeScoreVec* node_score_vec = context->last_layer_node_scores();
  uint32_t node_score_size = context->last_layer_node_score_size();

  LOG_DEBUG << "last layer total size: " << node_score_size
            << ", search_topn: " << search_topn;

  for (uint32_t i = 0; i < node_score_size && i < search_topn; ++i) {
    NodeScore* node_score = node_score_vec->at(i);
    UINode* node_info = node_score->node()->node_info();
    ResultUnit* unit = search_result->add_result_unit();
    unit->set_id(node_info->id());
    unit->set_score(node_score->score());
  }
  return true;
}

// register itself
REGISTER_INDEX(tree_index, TreeIndex);

}  // namespace tdm_serving
