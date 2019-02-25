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

#include "test/model/mock_model.h"
#include "model/predict_interface.h"
#include "util/log.h"

namespace tdm_serving {

MockModel::MockModel() {
}

MockModel::~MockModel() {
}

bool MockModel::Init(const ModelConf* model_conf) {
  mock_model_conf_ = model_conf;

  const std::string& section = mock_model_conf_->section();

  if (!Model::Init(model_conf)) {
    LOG_ERROR << "[" << section << "] Model::Init failed";
    return false;
  }

  return true;
}

bool MockModel::Predict(PredictContext* /*predict_ctx*/,
                       const PredictRequest& /*predict_req*/,
                       PredictResponse* predict_res) const {
  predict_res->add_score(0.5);
  return true;
}

PredictContext* MockModel::GetPredictContext() {
  return NULL;
}

void MockModel::ReleasePredictContext(PredictContext* /*context*/) {
  return;
}

ModelConf* MockModel::CreateModelConf() {
  return new ModelConf();
}

// register itself
REGISTER_MODEL(mock_model, MockModel);

}  // namespace tdm_serving
