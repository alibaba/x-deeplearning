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

#include "model/model_conf.h"

namespace tdm_serving {

TEST(ModelConf, init_no_version) {
  std::string conf_path = "test_data/conf/model.conf";
  std::string section_name = "test_model_conf_no_version";

  util::ConfParser conf_parser;
  ASSERT_TRUE(conf_parser.Init(conf_path));

  ModelConf model_conf;
  ASSERT_TRUE(model_conf.Init(section_name, conf_parser));

  EXPECT_EQ("test_model_conf_no_version", model_conf.section());
  EXPECT_EQ("test_data/model/mock_model_no_version", model_conf.model_path());
  EXPECT_EQ("test_data/model/mock_model_no_version",
            model_conf.latest_model_path());
  EXPECT_EQ("", model_conf.version_file_path());
  EXPECT_EQ("", model_conf.model_version());
}

TEST(ModelConf, init) {
  std::string conf_path = "test_data/conf/model.conf";
  std::string section_name = "test_model_conf";

  util::ConfParser conf_parser;
  ASSERT_TRUE(conf_parser.Init(conf_path));

  ModelConf model_conf;
  ASSERT_TRUE(model_conf.Init(section_name, conf_parser));

  EXPECT_EQ("test_model_conf", model_conf.section());
  EXPECT_EQ("test_data/model/mock_model", model_conf.model_path());
  EXPECT_EQ("test_data/model/mock_model/123458",
            model_conf.latest_model_path());
  EXPECT_EQ("test_data/model/mock_model/version",
            model_conf.version_file_path());
  EXPECT_EQ("123458", model_conf.model_version());
}

}  // namespace tdm_serving
