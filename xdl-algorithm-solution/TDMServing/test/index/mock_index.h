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

#ifndef TEST_INDEX_MOCK_INDEX_H_
#define TEST_INDEX_MOCK_INDEX_H_

#include <fstream>
#include "index/index.h"

namespace tdm_serving {

class IndexConf;

class MockIndex : public Index {
 public:
  MockIndex();
  virtual ~MockIndex();

  virtual bool Init(const IndexConf* index_conf);

  virtual bool Search(SearchContext* context,
                      const SearchParam& search_param,
                      SearchResult* search_result);

  virtual SearchContext* GetSearchContext();

  virtual void ReleaseSearchContext(SearchContext* context);

 protected:
  virtual IndexConf* CreateIndexConf();

 private:
  const IndexConf* mock_index_conf_;

  DISALLOW_COPY_AND_ASSIGN(MockIndex);
};

}  // namespace tdm_serving

#endif  // TEST_INDEX_MOCK_INDEX_H_
