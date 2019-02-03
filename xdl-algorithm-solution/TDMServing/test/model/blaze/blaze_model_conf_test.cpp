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

#include "model/blaze/blaze_model_conf.h"

namespace tdm_serving {

TEST(BlazeModelConf, init) {
  std::string conf_path = "test_data/conf/model.conf";
  std::string section_name = "simp_model";

  util::ConfParser conf_parser;
  ASSERT_TRUE(conf_parser.Init(conf_path));

  BlazeModelConf model_conf;
  ASSERT_TRUE(model_conf.Init(section_name, conf_parser));

  EXPECT_EQ(blaze::kPDT_CPU, model_conf.device_type());
}

}  // namespace tdm_serving
