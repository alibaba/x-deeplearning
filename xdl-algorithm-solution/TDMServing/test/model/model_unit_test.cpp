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

#include <unistd.h>
#include "gtest/gtest.h"

#define protected public
#define private public

#include "test/model/mock_model.h"
#include "model/model_unit.h"

namespace tdm_serving {

TEST(ModelUnit, init_disable) {
  std::string conf_path = "test_data/conf/model.conf";
  std::string section_name = "mock_model_disable";

  ModelUnit model_unit;
  ASSERT_TRUE(model_unit.Init(section_name, conf_path));

  EXPECT_FALSE(model_unit.is_enabled());
}

TEST(ModelUnit, init_no_version) {
  std::string conf_path = "test_data/conf/model.conf";
  std::string section_name = "mock_model_no_version";

  ModelUnit model_unit;
  ASSERT_TRUE(model_unit.Init(section_name, conf_path));

  EXPECT_TRUE(model_unit.is_enabled());
  EXPECT_EQ("", model_unit.GetModel()->version_file_path());
}

TEST(ModelUnit, init) {
  std::string conf_path = "test_data/conf/model.conf";
  std::string section_name = "mock_model";

  ModelUnit model_unit;
  ASSERT_TRUE(model_unit.Init(section_name, conf_path));

  EXPECT_TRUE(model_unit.is_enabled());
  EXPECT_EQ("test_data/model/mock_model/version",
            model_unit.GetModel()->version_file_path());
  EXPECT_EQ("test_data/model/mock_model",
            model_unit.GetModel()->model_conf_->model_path());
  EXPECT_EQ("test_data/model/mock_model/123458",
            model_unit.GetModel()->model_conf_->latest_model_path());
  EXPECT_EQ("123458",
            model_unit.GetModel()->model_conf_->model_version());
}

TEST(ModelUnit, reload) {
  std::string conf_path = "test_data/conf/model.conf";
  std::string section_name = "mock_model";

  ModelUnit model_unit;
  ASSERT_TRUE(model_unit.Init(section_name, conf_path));

  EXPECT_TRUE(model_unit.is_enabled());
  EXPECT_EQ("test_data/model/mock_model/version",
            model_unit.GetModel()->version_file_path());
  EXPECT_EQ("test_data/model/mock_model",
            model_unit.GetModel()->model_conf_->model_path());
  EXPECT_EQ("test_data/model/mock_model/123458",
            model_unit.GetModel()->model_conf_->latest_model_path());
  EXPECT_EQ("123458",
            model_unit.GetModel()->model_conf_->model_version());

  std::string model_path = model_unit.GetModel()->model_conf_->model_path();
  std::string version_path = model_path + "/version";
  std::string tmp_version_path = model_path + "/version.tmp";
  std::string new_version_path = model_path + "/version.new";

  std::string cmd = "cp " + version_path + " " + tmp_version_path;
  system(cmd.c_str());
  cmd = "cp " + new_version_path + " " + version_path;
  system(cmd.c_str());

  usleep(2000);

  EXPECT_EQ("test_data/model/mock_model/223458",
            model_unit.GetModel()->model_conf_->latest_model_path());
  EXPECT_EQ("223458",
            model_unit.GetModel()->model_conf_->model_version());

  cmd = "cp " + tmp_version_path + " " + version_path;
  system(cmd.c_str());
  cmd = "rm " + tmp_version_path;
  system(cmd.c_str());
}

}  // namespace tdm_serving
