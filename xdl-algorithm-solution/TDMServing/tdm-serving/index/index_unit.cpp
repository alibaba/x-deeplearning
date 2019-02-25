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

#include "index/index_unit.h"
#include <utime.h>
#include <fstream>
#include "common/common_def.h"
#include "index/index.h"
#include "model/model_manager.h"
#include "util/file_monitor.h"
#include "util/log.h"

namespace tdm_serving {

IndexUnit::IndexUnit() : enable_(true) {
}

IndexUnit::~IndexUnit() {
  for (uint32_t i = 0; i < index_datas_.size(); ++i) {
    DELETE_AND_SET_NULL(index_datas_[i]);
  }
  index_datas_.clear();
  if (!version_file_path_.empty()) {
    util::FileMonitor::UnWatch(version_file_path_);
  }
}

void IndexReloadAction(const std::string& file,
                       util::WatchEvent ev,
                       void* args) {
  (void)file;
  (void)ev;

  if (NULL == args) {
    LOG_ERROR << "Relaod action args is NULL";
    return;
  }

  IndexUnit* index_unit = reinterpret_cast<IndexUnit*>(args);
  if (NULL == index_unit) {
    LOG_ERROR << "Index Unit ptr is NULL";
    return;
  }

  // wait for model to update
  if (!index_unit->CheckModelVersion()) {
    LOG_WARN << "Index Unit check model version failed";
    utime(file.c_str(), NULL);
    sleep(1);
    return;
  }

  if (!index_unit->Reload()) {
    LOG_ERROR << "Index Unit reload failed";
  }
}

bool IndexUnit::Init(const std::string& section,
                     const std::string& conf_path) {
  section_ = section;
  conf_path_ = conf_path;

  if (!conf_parser_.Init(conf_path_)) {
    LOG_ERROR << "Load conf from " << conf_path << " failed";
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

  if (!conf_parser_.GetValue<std::string>(section_, kConfigIndexType,
                                          &index_type_)
      || index_type_.empty()) {
    LOG_ERROR <<
        "[" << section_ << "] get " << kConfigIndexType << " failed";
    return false;
  }
  LOG_INFO << "[" << section_ << "] "
                << kConfigIndexType << ": "<< index_type_;

  for (uint32_t i = 0; i < kIndexInstanceNum; ++i) {
    index_datas_.push_back(NULL);
  }

  if (!Reload()) {
    LOG_ERROR << "init unit init to reload failed";
    return false;
  }

  // register file monitor
  version_file_path_ = index_datas_[idx_]->version_file_path();
  if (version_file_path_.empty()) {
    LOG_INFO << "[" << section_ << "] need not reload index";
  } else {
    util::FileMonitor::Watch(version_file_path_, IndexReloadAction,
                             reinterpret_cast<bool *>(this));
    LOG_INFO << "[" << section_ << "] file_monitor register at: "
                  << version_file_path_;
  }

  return true;
}

bool IndexUnit::Reload() {
  LOG_INFO << "[" << section_ << "] begin reload";

  Index* index = IndexRegisterer::GetInstanceByTitle(index_type_);
  if (index == NULL) {
    LOG_ERROR <<
        "[" << section_ << "] get index by type: " << index_type_ << " failed";
    return false;
  }
  if (!index->Init(section_, conf_parser_)) {
    LOG_ERROR << "[" << section_ << "] init index failed";
    return false;
  }

  // switch
  uint32_t new_idx = (idx_ + 1) % kIndexInstanceNum;
  Index* old_index = index_datas_[new_idx];
  index_datas_[new_idx] = index;
  idx_ = new_idx;
  if (old_index != NULL) {
    LOG_INFO << "[" << section_ << "] release old index";
    delete old_index;
  }
  LOG_INFO << "[" << section_ << "] new index switch success";

  return true;
}

Index* IndexUnit::GetIndex() {
  return index_datas_[idx_];
}

bool IndexUnit::is_enabled() {
  return enable_;
}

bool IndexUnit::CheckModelVersion() {
  std::ifstream version_file_handler(version_file_path_.c_str());
  if (!version_file_handler) {
    LOG_DEBUG << "Check model version, open "
                   << version_file_path_ << " failed, need no check";
    return true;
  }

  std::string model_name = GetIndex()->model_name();
  if (model_name.empty()) {
    LOG_DEBUG << "Check model version, model name is empty, "
                   << "do not need model";
    return true;
  }

  // get model version for index
  std::string line;
  size_t pos = 0;
  std::string model_version;

  while (std::getline(version_file_handler, line)) {
    util::StrUtil::Trim(line);
    if ((pos = line.find(kModelVersionTag)) != std::string::npos) {
      model_version = line.substr(pos + kModelVersionTag.length());
      util::StrUtil::Trim(model_version);
    }
  }
  if (model_version == "") {
    LOG_ERROR << "Check model version, has version file, "
                   << "but has no version, need no check";
    return true;
  }

  // check if model version is valid
  if (!ModelManager::Instance().HasModel(model_name, model_version)) {
    LOG_WARN << "Check model version, "
                  << "model manager does not have model with "
                  << "model_name: " << model_name
                  << ", model_version: " << model_version
                  << ", wait for model updating";
    return false;
  }

  LOG_DEBUG << "Check model version, "
                 << "model manager has model with "
                 << "model_name: " << model_name
                 << ", model_version: " << model_version
                 << ", pass";
  return true;
}

}  // namespace tdm_serving
