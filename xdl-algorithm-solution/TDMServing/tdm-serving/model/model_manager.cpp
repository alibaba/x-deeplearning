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

#include "model/model_manager.h"
#include "model/model_unit.h"
#include "model/model.h"
#include "model/predict_context.h"
#include "model/predict_interface.h"
#include "blaze/include/predictor.h"
#include "util/log.h"

namespace tdm_serving {

ModelManager::ModelManager() {
  inited_ = false;
}

ModelManager::~ModelManager() {
  Reset();
}

void ModelManager::Reset() {
  ModelMap::iterator iter = model_map_.begin();
  for (; iter != model_map_.end(); ++iter) {
    DELETE_AND_SET_NULL(iter->second);
  }
  inited_ = false;
}

bool ModelManager::Init(const std::string& conf_path) {
  if (inited_) {
    return true;
  }

  util::SimpleMutex::Locker slock(&mutex_);
  if (inited_) {
    return true;
  }

  // blaze init scheduler
  blaze::InitScheduler(false, 1000, 100, 32, 4, 2);

  util::ConfParser conf_parser;
  if (!conf_parser.Init(conf_path)) {
    LOG_ERROR <<
        "Model Manager load conf from [" << conf_path << "] failed";
    return false;
  }

  const std::vector<util::ConfSection*>& conf_sections =
      conf_parser.GetAllConfSection();

  for (uint32_t i = 0; i < conf_sections.size(); ++i) {
    // create each data_unit
    const util::ConfSection* conf_section = conf_sections[i];
    if (conf_section == NULL) {
      LOG_ERROR << "init model_manager failed, NULL conf_section";
      return false;
    }

    const std::string& section = conf_section->GetSectionName();
    ModelUnit* model_unit = new ModelUnit();
    if (!model_unit->Init(section, conf_path)) {
      LOG_ERROR << "[" << section << "] init model unit failed";
      return false;
    }

    if (!model_unit->is_enabled()) {
      delete model_unit;
    } else {
      model_map_[section] = model_unit;
      LOG_INFO << "[" << section << "] init model unit success";
    }
  }

  inited_ = true;

  return true;
}

bool ModelManager::Predict(
    PredictContext* predict_ctx,
    const PredictRequest& predict_req,
    PredictResponse* predict_res) {
  Model* model = GetModel(predict_req.model_name(),
                          predict_req.model_version());

  if (model == NULL) {
    LOG_WARN << "get NULL model by model_name: "
                  << predict_req.model_name()
                  << ", model_version: "
                  << predict_req.model_version();
    return false;
  }

  if (!model->Predict(predict_ctx, predict_req, predict_res)) {
    LOG_WARN << "do search with model name: "
                  << predict_req.model_name()
                  << ", model_version: "
                  << predict_req.model_version()
                  << " failed";
    return false;
  }
  return true;
}

PredictContext*
ModelManager::GetPredictContext(const std::string& model_name) {
  Model* model = GetModel(model_name);
  if (model == NULL) {
    LOG_DEBUG << "get predict context failed";
    return NULL;
  } else {
    return model->GetPredictContext();
  }
}

void ModelManager::ReleasePredictContext(const std::string& model_name,
                                         PredictContext* context) {
  Model* model = GetModel(model_name);
  if (model != NULL) {
    model->ReleasePredictContext(context);
  } else {
    LOG_DEBUG << "release predict context failed";
  }
}

ModelUnit* ModelManager::GetModelUnit(const std::string& model_name) {
  ModelMap::iterator iter = model_map_.find(model_name);
  if (iter == model_map_.end()) {
    LOG_WARN << "model unit with model_name: "
                  << model_name << " is not in model map";
    return NULL;
  }
  return iter->second;
}

Model* ModelManager::GetModel(const std::string& model_name,
                              const std::string& model_version) {
  ModelUnit* model_unit = GetModelUnit(model_name);
  if (model_unit == NULL) {
    LOG_WARN << "get model unit by model_name: "
                  << model_name << " failed";
    return NULL;
  }

  Model* model = model_unit->GetModel(model_version);
  if (model == NULL) {
    LOG_WARN << "get index by model unit with model_name: "
                  << model_name << " failed";
  }
  return model;
}

bool ModelManager::HasModel(const std::string& model_name,
                            const std::string& model_version) {
  Model* model = GetModel(model_name, model_version);
  if (model == NULL) {
    return false;
  }
  return true;
}

}  // namespace tdm_serving
