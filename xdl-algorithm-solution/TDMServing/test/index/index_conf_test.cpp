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

#include "index/index_conf.h"

namespace tdm_serving {

TEST(IndexConf, init_no_version) {
  std::string conf_path = "test_data/conf/index.conf";
  std::string section_name = "test_index_conf_no_version";

  util::ConfParser conf_parser;
  ASSERT_TRUE(conf_parser.Init(conf_path));

  IndexConf index_conf;
  ASSERT_TRUE(index_conf.Init(section_name, conf_parser));

  EXPECT_EQ("test_index_conf_no_version", index_conf.section());
  EXPECT_EQ("test_data/index/mock_index_no_version", index_conf.index_path());
  EXPECT_EQ("test_data/index/mock_index_no_version",
            index_conf.latest_index_path());
  EXPECT_EQ("", index_conf.version_file_path());
  EXPECT_EQ("", index_conf.index_version());
  EXPECT_EQ("mock_model", index_conf.model_name());
  EXPECT_EQ("", index_conf.model_version());
  EXPECT_EQ(2, index_conf.build_omp());
}

TEST(IndexConf, init) {
  std::string conf_path = "test_data/conf/index.conf";
  std::string section_name = "test_index_conf";

  util::ConfParser conf_parser;
  ASSERT_TRUE(conf_parser.Init(conf_path));

  IndexConf index_conf;
  ASSERT_TRUE(index_conf.Init(section_name, conf_parser));

  EXPECT_EQ("test_index_conf", index_conf.section());
  EXPECT_EQ("test_data/index/mock_index", index_conf.index_path());
  EXPECT_EQ("test_data/index/mock_index/123456",
            index_conf.latest_index_path());
  EXPECT_EQ("test_data/index/mock_index/version",
            index_conf.version_file_path());
  EXPECT_EQ("123456", index_conf.index_version());
  EXPECT_EQ("mock_model", index_conf.model_name());
  EXPECT_EQ("123458", index_conf.model_version());
  EXPECT_EQ(2, index_conf.build_omp());
}

}  // namespace tdm_serving
