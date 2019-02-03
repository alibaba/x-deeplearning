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

#include "model/model_unit.h"
#include "common/common_def.h"
#include "model/model.h"
#include "util/file_monitor.h"
#include "util/log.h"

namespace tdm_serving {

ModelUnit::ModelUnit() : enable_(true) {
}

ModelUnit::~ModelUnit() {
  for (uint32_t i = 0; i < model_datas_.size(); ++i) {
    DELETE_AND_SET_NULL(model_datas_[i]);
  }
  model_datas_.clear();
  if (!version_file_path_.empty()) {
    util::FileMonitor::UnWatch(version_file_path_);
  }
}

void ModelReloadAction(const std::string& file,
                       util::WatchEvent ev,
                       void* args) {
  (void)file;
  (void)ev;
  if (NULL == args) {
    LOG_ERROR << "relaod action args is NULL";
    return;
  }
  ModelUnit* model_unit = reinterpret_cast<ModelUnit*>(args);
  if (NULL == model_unit) {
    LOG_ERROR << "ModelUnit ptr is NULL";
    return;
  }
  if (!model_unit->Reload()) {
    LOG_ERROR << "model unit reload failed";
  }
}

bool ModelUnit::Init(const std::string& section,
                     const std::string& conf_path) {
  section_ = section;
  conf_path_ = conf_path;

  if (!conf_parser_.Init(conf_path_)) {
    LOG_ERROR << "load conf from " << conf_path_ << " failed";
    return false;
  }

  if (!conf_parser_.GetValue<bool>(section_, kConfigEnable, &enable_)) {
    LOG_ERROR << "[" << section_ << "] get "
                   << kConfigEnable << " failed";
    return false;
  }

  if (!enable_) {
    LOG_INFO << "[" << section_ << "] is disable";
    return true;
  }

  if (!conf_parser_.GetValue<std::string>(section_, kConfigModelType,
                                          &model_type_)
      || model_type_.empty()) {
    LOG_ERROR <<
        "[" << section_ << "] get " << kConfigModelType << " failed";
    return false;
  }
  LOG_INFO << "[" << section_ << "] "
                << kConfigModelType << ": " << model_type_;

  for (uint32_t i = 0; i < kModelInstanceNum; ++i) {
    model_datas_.push_back(NULL);
  }

  if (!Reload()) {
    LOG_ERROR << "model unit init to reload failed";
    return false;
  }

  // register file monitor
  version_file_path_ = model_datas_[idx_]->version_file_path();
  if (version_file_path_.empty()) {
    LOG_INFO << "[" << section_ << "] need not reload model";
  } else {
    util::FileMonitor::Watch(version_file_path_, ModelReloadAction,
                             reinterpret_cast<bool *>(this));
    LOG_INFO << "[" << section_ << "] file_monitor register at: "
                  << version_file_path_;
  }

  return true;
}

bool ModelUnit::Reload() {
  LOG_INFO << "[" << section_ << "] begin reload";

  Model* model = ModelRegisterer::GetInstanceByTitle(model_type_);
  if (model == NULL) {
    LOG_ERROR <<
        "[" << section_ << "] get model_type: " << model_type_ << " failed";
    return false;
  }
  if (!model->Init(section_, conf_parser_)) {
    LOG_ERROR << "[" << section_ << "] init model failed";
    return false;
  }

  // switch
  uint32_t new_idx = (idx_ + 1) % kModelInstanceNum;
  Model* old_model = model_datas_[new_idx];
  model_datas_[new_idx] = model;
  idx_ = new_idx;
  if (old_model != NULL) {
    LOG_INFO << "[" << section_ << "] release old model";
    delete old_model;
  }
  LOG_INFO << "[" << section_ << "] new model switch success";

  return true;
}

bool ModelUnit::is_enabled() {
  return enable_;
}

Model* ModelUnit::GetModel(const std::string& model_version) {
  if (model_version.empty()) {
    return model_datas_[idx_];
  }

  for (size_t i = 0; i < model_datas_.size(); i++) {
    if (model_datas_[i] != NULL &&
        model_datas_[i]->model_version() == model_version) {
      return model_datas_[i];
    }
  }
  return NULL;
}

}  // namespace tdm_serving
