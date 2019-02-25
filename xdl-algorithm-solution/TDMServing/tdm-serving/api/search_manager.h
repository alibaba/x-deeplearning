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

#ifndef TDM_SERVING_API_SEARCH_MANAGER_H_
#define TDM_SERVING_API_SEARCH_MANAGER_H_

#include <string>

namespace tdm_serving {

class SearchParam;
class SearchResult;

// User interface
class SearchManager {
 public:
  SearchManager();
  ~SearchManager();

  static SearchManager& Instance() {
    static SearchManager search_manager;
    return search_manager;
  }

  // Initialize index and model by config file,
  // @param index_conf_path: index conf file path
  // @param model_conf_path: model conf file path
  // @param filter_conf_path: filter conf file path
  // @Return True: success False: failed
  bool Init(const std::string& index_conf_path,
            const std::string& model_conf_path,
            const std::string& filter_conf_path = "");

  // Search interface
  // @param search_param: search request
  // @param search_result: search response
  bool Search(const SearchParam& search_param,
              SearchResult* search_result);
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_API_SEARCH_MANAGER_H_
