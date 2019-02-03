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

#include "test/index/mock_index.h"
#include "index/search_context.h"
#include "util/object_free_list.h"
#include "util/log.h"

namespace tdm_serving {

MockIndex::MockIndex() {
}

MockIndex::~MockIndex() {
}

bool MockIndex::Init(const IndexConf* index_conf) {
  mock_index_conf_ = index_conf;

  const std::string& section = mock_index_conf_->section();

  if (!Index::Init(index_conf)) {
    LOG_ERROR << "[" << section << "] Index::Init failed";
    return false;
  }

  return true;
}

bool MockIndex::Search(SearchContext* /*context*/,
                       const SearchParam& /*search_param*/,
                       SearchResult* /*search_result*/) {
  return true;
}

SearchContext* MockIndex::GetSearchContext() {
  return util::ObjList<SearchContext>::Instance().Get();
}

void MockIndex::ReleaseSearchContext(SearchContext* context) {
  util::ObjList<SearchContext>::Instance().Free(
      static_cast<SearchContext*>(context));
}

IndexConf* MockIndex::CreateIndexConf() {
  return new IndexConf();
}

// register itself
REGISTER_INDEX(mock_index, MockIndex);

}  // namespace tdm_serving
