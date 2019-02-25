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

#include "api/search_manager.h"
#include "index/index_manager.h"
#include "model/model_manager.h"
#include "biz/filter_manager.h"
#include "util/log.h"

namespace tdm_serving {

SearchManager::SearchManager() {
}

SearchManager::~SearchManager() {
}

bool SearchManager::Init(const std::string& index_conf_path,
                         const std::string& model_conf_path,
                         const std::string& filter_conf_path) {
  bool ret = false;

  if (!filter_conf_path.empty()) {
    ret = FilterManager::Instance().Init(filter_conf_path);
    if (ret == false) {
      LOG_ERROR << "Filter Manager init failed";
      return false;
    }
  }

  ret = ModelManager::Instance().Init(model_conf_path);
  if (ret == false) {
    LOG_ERROR << "Model Manager init failed";
    return false;
  }

  ret = IndexManager::Instance().Init(index_conf_path);
  if (ret == false) {
    LOG_ERROR << "Index Manager init failed";
    return false;
  }

  return true;
}

bool SearchManager::Search(const SearchParam& search_param,
                           SearchResult* search_result) {
  if (search_result == NULL) {
    LOG_ERROR << "search_result is NULL";
    return false;
  }

  search_result->set_res_code(RC_SUCCESS);

  if (!IndexManager::Instance().Search(search_param, search_result)) {
    LOG_ERROR << "Index Manager search failed";
    search_result->set_res_code(RC_SEARCH_ERROR);
    return false;
  }

  return true;
}

}  // namespace tdm_serving
