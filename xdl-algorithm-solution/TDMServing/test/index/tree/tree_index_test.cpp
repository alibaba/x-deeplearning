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

#include "gtest/gtest.h"

#define protected public
#define private public

#include "test/index/tree/tree_test_util.h"
#include "index/tree/tree_index.h"
#include "index/tree/tree_index_conf.h"
#include "index/tree/tree_search_context.h"

namespace tdm_serving {

class MockTreeSearcher2 : public TreeSearcher {
 protected:
  virtual bool CalculateScore(TreeSearchContext* /*context*/,
                              const SearchParam& /*search_param*/,
                              std::vector<ItemFeature*>* /*item_features*/,
                              std::vector<NodeScore*>* node_scores) {
    for (size_t i = 0; i < node_scores->size(); i++) {
      node_scores->at(i)->set_score(i * 10);
    }
    return true;
  }
};

class MockTreeIndex : public TreeIndex {
 public:
  virtual bool Init(const IndexConf* index_conf) {
    if (!TreeIndex::Init(index_conf)) {
      return false;
    }
    if (!mock_tree_searcher_.Init(tree_index_conf_)) {
      return false;
    }
    return true;
  }

 protected:
  virtual TreeSearcher* tree_searcher() {
    return &mock_tree_searcher_;
  }
 private:
  MockTreeSearcher2 mock_tree_searcher_;
};

TEST(TreeIndex, init_and_search) {
  std::string conf_path = "test_data/conf/index.conf";
  std::string section_name = "tree_index";

  // init pb tree
  ASSERT_TRUE(CreateTestTreeIndex("test_data/index/tree_index"));

  // init index
  util::ConfParser conf_parser;
  ASSERT_TRUE(conf_parser.Init(conf_path));

  Index* index = new MockTreeIndex;
  ASSERT_TRUE(index->Init(section_name, conf_parser));

  // search
  TreeSearchContext search_ctx;

  SearchParam search_param;
  search_param.set_index_name(section_name);
  search_param.set_topn(2);

  SearchResult search_result;
  ASSERT_TRUE(index->Prepare(&search_ctx, search_param));
  ASSERT_TRUE(index->Search(&search_ctx, search_param, &search_result));

  ASSERT_EQ(search_result.result_unit_size(), 2);
  ASSERT_EQ(search_result.result_unit(0).id(), 121llu);
  ASSERT_FLOAT_EQ(search_result.result_unit(0).score(), 30.0);
  ASSERT_EQ(search_result.result_unit(1).id(), 111llu);
  ASSERT_FLOAT_EQ(search_result.result_unit(1).score(), 20.0);
}

}  // namespace tdm_serving
