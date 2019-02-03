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

#include "model/model_manager.h"
#include "model/model.h"
#include "model/predict_interface.h"

namespace tdm_serving {

TEST(ModelManager, init_and_search) {
  std::string conf_path = "test_data/conf/model_manager.conf";

  // init
  ASSERT_TRUE(ModelManager::Instance().Init(conf_path));

  Model* model = ModelManager::Instance().GetModel("mock_model_disable");
  EXPECT_STREQ(nullptr, reinterpret_cast<const char*>(model));

  model = ModelManager::Instance().GetModel("mock_model_no_version");
  EXPECT_EQ("mock_model_no_version", model->model_name());

  model = ModelManager::Instance().GetModel("mock_model");
  EXPECT_EQ("mock_model", model->model_name());

  // predict
  PredictRequest predict_request;
  predict_request.set_model_name("mock_model");

  PredictResponse predict_response;

  ASSERT_TRUE(ModelManager::Instance().Predict(NULL,
                                               predict_request,
                                               &predict_response));
  ASSERT_EQ(1, predict_response.score_size());
  EXPECT_FLOAT_EQ(0.5, predict_response.score(0));
}

}  // namespace tdm_serving
