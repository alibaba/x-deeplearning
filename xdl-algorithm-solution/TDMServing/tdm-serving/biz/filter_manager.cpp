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

#include "biz/filter_manager.h"
#include "common/common_def.h"
#include "biz/filter.h"
#include "util/str_util.h"
#include "util/log.h"

namespace tdm_serving {

FilterManager::FilterManager() {
  inited_ = false;
}

FilterManager::~FilterManager() {
  Reset();
}

void FilterManager::Reset() {
  FilterMap::iterator iter = filter_map_.begin();
  for (; iter != filter_map_.end(); ++iter) {
    DELETE_AND_SET_NULL(iter->second);
  }
  inited_ = false;
}

bool FilterManager::Init(const std::string& conf_path) {
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
    // create each filter
    const util::ConfSection* conf_section = conf_sections[i];
    if (conf_section == NULL) {
      LOG_ERROR << "Filter Manager init failed, NULL conf_section";
      return false;
    }

    const std::string& section = conf_section->GetSectionName();

    std::string filter_type;
    if (!conf_parser.GetValue<std::string>(section, kConfigFilterType,
                                           &filter_type)
        || filter_type.empty()) {
      LOG_ERROR << "[" << section << "] get filter by type: "
                << filter_type << " failed";
      return false;
    }
    LOG_INFO << "[" << section << "] "
                << kConfigFilterType << ":" << filter_type;

    // get filter by type
    Filter* filter = FilterRegisterer::GetInstanceByTitle(filter_type);
    if (filter == NULL) {
      LOG_ERROR << "[" << section << "] get filter by type: "
                << filter_type << " failed";
      return false;
    }

    // init filter
    if (!filter->Init(section, conf_parser)) {
      LOG_ERROR << "[" << section << "] Index Unit init failed";
      return false;
    }

    filter_map_[section] = filter;
  }

  inited_ = true;

  return true;
}

Filter* FilterManager::GetFilter(const std::string& filter_name) {
  FilterMap::iterator iter = filter_map_.find(filter_name);
  if (iter == filter_map_.end()) {
    LOG_WARN << "Filter with filter_name: "
             << filter_name << " is not in filter map";
    return NULL;
  }
  return iter->second;
}

}  // namespace tdm_serving
