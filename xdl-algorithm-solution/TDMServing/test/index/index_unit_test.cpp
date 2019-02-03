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
#include <utime.h>
#include <thread>
#include "gtest/gtest.h"

#define protected public
#define private public

#include "test/index/mock_index.h"
#include "index/index_unit.h"
#include "model/model_manager.h"
#include "util/log.h"

namespace tdm_serving {

TEST(IndexUnit, init_disable) {
  std::string conf_path = "test_data/conf/index.conf";
  std::string section_name = "mock_index_disable";

  IndexUnit index_unit;
  ASSERT_TRUE(index_unit.Init(section_name, conf_path));

  EXPECT_FALSE(index_unit.is_enabled());
}

TEST(IndexUnit, init_no_version) {
  std::string conf_path = "test_data/conf/index.conf";
  std::string section_name = "mock_index_no_version";

  IndexUnit index_unit;
  ASSERT_TRUE(index_unit.Init(section_name, conf_path));

  EXPECT_TRUE(index_unit.is_enabled());
  EXPECT_EQ("", index_unit.GetIndex()->version_file_path());
}

TEST(IndexUnit, init) {
  std::string conf_path = "test_data/conf/index.conf";
  std::string section_name = "mock_index";

  IndexUnit index_unit;
  ASSERT_TRUE(index_unit.Init(section_name, conf_path));

  EXPECT_TRUE(index_unit.is_enabled());
  EXPECT_EQ("test_data/index/mock_index/version",
            index_unit.GetIndex()->version_file_path());
  EXPECT_EQ("test_data/index/mock_index",
            index_unit.GetIndex()->index_conf_->index_path());
  EXPECT_EQ("test_data/index/mock_index/123456",
            index_unit.GetIndex()->index_conf_->latest_index_path());
  EXPECT_EQ("123456",
            index_unit.GetIndex()->index_conf_->index_version());
}

TEST(IndexUnit, reload) {
  std::string conf_path = "test_data/conf/index.conf";
  std::string section_name = "mock_index";

  IndexUnit index_unit;
  ASSERT_TRUE(index_unit.Init(section_name, conf_path));

  EXPECT_TRUE(index_unit.is_enabled());
  EXPECT_EQ("test_data/index/mock_index/version",
            index_unit.GetIndex()->version_file_path());
  EXPECT_EQ("test_data/index/mock_index",
            index_unit.GetIndex()->index_conf_->index_path());
  EXPECT_EQ("test_data/index/mock_index/123456",
            index_unit.GetIndex()->index_conf_->latest_index_path());
  EXPECT_EQ("123456",
            index_unit.GetIndex()->index_conf_->index_version());

  std::string index_path = index_unit.GetIndex()->index_conf_->index_path();
  std::string version_path = index_path + "/version";
  std::string tmp_version_path = index_path + "/version.tmp";
  std::string new_version_path = index_path + "/version.new";

  std::string cmd = "cp " + version_path + " " + tmp_version_path;
  system(cmd.c_str());
  cmd = "cp " + new_version_path + " " + version_path;
  system(cmd.c_str());

  usleep(5000);

  EXPECT_EQ("test_data/index/mock_index/223456",
            index_unit.GetIndex()->index_conf_->latest_index_path());
  EXPECT_EQ("223456",
            index_unit.GetIndex()->index_conf_->index_version());

  cmd = "cp " + tmp_version_path + " " + version_path;
  system(cmd.c_str());
  cmd = "rm " + tmp_version_path;
  system(cmd.c_str());
}

}  // namespace tdm_serving
