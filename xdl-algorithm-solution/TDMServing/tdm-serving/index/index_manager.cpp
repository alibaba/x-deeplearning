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

#include "index/index_manager.h"
#include "common/common_def.h"
#include "index/index_unit.h"
#include "index/index.h"
#include "index/search_context.h"
#include "model/model_manager.h"
#include "model/model.h"
#include "model/predict_context.h"
#include "util/log.h"

namespace tdm_serving {

IndexManager::IndexManager() {
  inited_ = false;
}

IndexManager::~IndexManager() {
  Reset();
}

void IndexManager::Reset() {
  IndexMap::iterator iter = index_map_.begin();
  for (; iter != index_map_.end(); ++iter) {
    DELETE_AND_SET_NULL(iter->second);
  }
  inited_ = false;
}

bool IndexManager::Init(const std::string& conf_path) {
  if (inited_) {
    return true;
  }
  util::SimpleMutex::Locker slock(&mutex_);
  if (inited_) {
    return true;
  }

  util::ConfParser conf_parser;
  if (!conf_parser.Init(conf_path)) {
    LOG_ERROR << "Index Manager load conf from "
                   << "[" << conf_path << "] failed";
    return false;
  }

  const std::vector<util::ConfSection*>& conf_sections =
      conf_parser.GetAllConfSection();

  for (uint32_t i = 0; i < conf_sections.size(); ++i) {
    // create each data_unit
    const util::ConfSection* conf_section = conf_sections[i];
    if (conf_section == NULL) {
      LOG_ERROR << "Index Manager init failed, NULL conf_section";
      return false;
    }

    const std::string& section = conf_section->GetSectionName();

    IndexUnit* index_unit = new IndexUnit();

    if (!index_unit->Init(section, conf_path)) {
      LOG_ERROR << "[" << section << "] Index Unit init failed";
      return false;
    }

    if (!index_unit->is_enabled()) {
      delete index_unit;
    } else {
      index_map_[section] = index_unit;
      LOG_INFO << "[" << section << "] Index Unit init success";
    }
  }

  inited_ = true;

  return true;
}

bool IndexManager::Search(const SearchParam& search_param,
                          SearchResult* search_result) {
  const std::string& index_name = search_param.index_name();

  Index* index = GetIndex(index_name);
  if (index == NULL) {
    LOG_WARN << "Get NULL index by index_name: " << index_name;
    return false;
  }

  SearchContext* search_ctx = index->GetSearchContext();
  if (search_ctx == NULL) {
    LOG_WARN << "Get NULL search context by index_name: " << index_name;
    return false;
  }

  if (!index->Prepare(search_ctx, search_param)) {
    LOG_WARN << "Prepare with index_name: " << index_name << " failed";
    index->ReleaseSearchContext(search_ctx);
    return false;
  }

  if (!index->Search(search_ctx, search_param, search_result)) {
    LOG_WARN << "Search with index_name: " << index_name << " failed";
    index->ReleaseSearchContext(search_ctx);
    return false;
  }

  index->ReleaseSearchContext(search_ctx);

  return true;
}

IndexUnit* IndexManager::GetIndexUnit(const std::string& index_name) {
  IndexMap::iterator iter = index_map_.find(index_name);
  if (iter == index_map_.end()) {
    LOG_WARN << "Index Unit with index_name: "
                  << index_name << " is not in index map";
    return NULL;
  }
  return iter->second;
}

Index* IndexManager::GetIndex(const std::string& index_name) {
  IndexUnit* index_unit = GetIndexUnit(index_name);
  if (index_unit == NULL) {
    LOG_WARN << "Get Index Unit by index_name: "
                  << index_name << " failed";
    return NULL;
  }
  Index* index = index_unit->GetIndex();
  if (index == NULL) {
    LOG_WARN << "Get Index by Index Unit with index_name: "
                  << index_name << " failed";
  }
  return index;
}

}  // namespace tdm_serving
