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

#ifndef TDM_SERVING_INDEX_INDEX_MANAGER_H_
#define TDM_SERVING_INDEX_INDEX_MANAGER_H_

#include <string>
#include "common/common_def.h"
#include "util/conf_parser.h"
#include "util/singleton.h"
#include "util/concurrency/mutex.h"
#include "proto/search.pb.h"

namespace tdm_serving {

class IndexUnit;
class Index;
class Filter;

// Index Manager manages all index instances,
// monitor index version file and reload index.
// It also provide search interface for all indexes
class IndexManager : public util::Singleton<IndexManager> {
 public:
  IndexManager();
  ~IndexManager();

  // Initialize by index conf file,
  bool Init(const std::string& conf_path);

  void Reset();

  // Search interface
  bool Search(const SearchParam& search_param,
              SearchResult* search_result);

  // Get index unit by index name
  IndexUnit* GetIndexUnit(const std::string& index_name);

  // Get index by name
  Index* GetIndex(const std::string& index_name);

 private:
  typedef std::map<std::string, IndexUnit*> IndexMap;
  IndexMap index_map_;

  util::SimpleMutex mutex_;
  bool inited_;

  DISALLOW_COPY_AND_ASSIGN(IndexManager);
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_INDEX_MANAGER_H_
