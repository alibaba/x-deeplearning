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

#include "index/tree/tree_index_conf.h"

namespace tdm_serving {

TEST(TreeIndexConf, init) {
  std::string conf_path = "test_data/conf/index.conf";
  std::string section_name = "test_tree_index_conf";

  util::ConfParser conf_parser;
  ASSERT_TRUE(conf_parser.Init(conf_path));

  TreeIndexConf index_conf;
  ASSERT_TRUE(index_conf.Init(section_name, conf_parser));

  EXPECT_EQ("meta_model", index_conf.model_name());
  EXPECT_EQ(512u, index_conf.tree_level_topn(8));
  EXPECT_EQ(200u, index_conf.tree_level_topn(11));
  EXPECT_EQ(300u, index_conf.tree_level_topn(13));
  EXPECT_EQ("100", index_conf.item_feature_group_id());
}

}  // namespace tdm_serving
