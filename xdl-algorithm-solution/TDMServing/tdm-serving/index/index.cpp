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

#include "index/index.h"
#include <algorithm>
#include "omp.h"
#include "index/search_context.h"
#include "biz/filter_manager.h"
#include "util/str_util.h"
#include "util/log.h"

namespace tdm_serving {

Index::Index()
    : index_conf_(NULL), filter_(NULL) {
}

Index::~Index() {
  DELETE_AND_SET_NULL(index_conf_);
}

bool Index::Init(const std::string& section,
                 const util::ConfParser& conf_parser) {
  index_conf_ = CreateIndexConf();
  index_conf_->Init(section, conf_parser);

  if (!Init(index_conf_)) {
    return false;
  }

  // set filter
  std::string filter_name = index_conf_->filter_name();
  if (!filter_name.empty()) {
    filter_ = FilterManager::Instance().GetFilter(filter_name);
    if (filter_ == NULL) {
      LOG_ERROR << "[" << section << "] get filter by name: "
                << filter_name << " failed";
      return false;
    }
  }

  return true;
}

IndexConf* Index::CreateIndexConf() {
  return new IndexConf();
}

bool Index::Init(const IndexConf* /*index_conf*/) {
  // set build omp
  omp_set_num_threads(index_conf_->build_omp());

  return true;
}

bool Index::Prepare(SearchContext* context,
                    const SearchParam& /*search_param*/) {
  // model name, used to get predict context
  context->set_model_name(index_conf_->model_name());

  // filter
  context->set_filter(filter_);

  return true;
}

}  // namespace tdm_serving
