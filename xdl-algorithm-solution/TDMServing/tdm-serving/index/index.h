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

#ifndef TDM_SERVING_INDEX_INDEX_H_
#define TDM_SERVING_INDEX_INDEX_H_

#include <string>
#include "common/common_def.h"
#include "index/index_conf.h"
#include "util/conf_parser.h"
#include "util/registerer.h"
#include "proto/search.pb.h"

namespace tdm_serving {

class SearchContext;
class Filter;

class Index {
 public:
  Index();
  virtual ~Index();

  // Initialize the index by parsed config,
  // section specifies the section name of index in config
  bool Init(const std::string& section,
            const util::ConfParser& conf_parser);

  // Initialize by parsed index config
  virtual bool Init(const IndexConf* index_conf);

  // Prepare for search
  virtual bool Prepare(SearchContext* context,
                       const SearchParam& search_param);

  // Do search
  virtual bool Search(SearchContext* context,
                      const SearchParam& search_param,
                      SearchResult* search_result) = 0;

  // Get sesssion data used for searching
  virtual SearchContext* GetSearchContext() = 0;

  // Release search context
  virtual void ReleaseSearchContext(SearchContext* context) = 0;

  // Get paht of version file used for index reloading
  const std::string& version_file_path() {
    return index_conf_->version_file_path();
  }

  // Get index_name
  const std::string& index_name() {
    return index_conf_->section();
  }

  // Get model_name
  const std::string& model_name() {
    return index_conf_->model_name();
  }

  // Set filter
  void set_filter(Filter* filter) {
    filter_ = filter;
  }

 protected:
  virtual IndexConf* CreateIndexConf();

 private:
  IndexConf* index_conf_;
  Filter* filter_;

  DISALLOW_COPY_AND_ASSIGN(Index);
};

// define register
REGISTER_REGISTERER(Index);
#define REGISTER_INDEX(title, name) \
    REGISTER_CLASS(Index, title, name)

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_INDEX_H_
