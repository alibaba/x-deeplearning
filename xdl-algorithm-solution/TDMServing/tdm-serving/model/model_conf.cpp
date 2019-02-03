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

#include "model/model_conf.h"
#include <fstream>
#include "common/common_def.h"
#include "util/str_util.h"
#include "util/log.h"

namespace tdm_serving {

ModelConf::ModelConf() {
}

ModelConf::~ModelConf() {
}

bool ModelConf::Init(const std::string& section,
                     const util::ConfParser& conf_parser) {
  section_ = section;

  // model_path
  if (!conf_parser.GetValue<std::string>(section, kConfigModelPath,
                                         &model_path_)
      || model_path_.empty()) {
    LOG_ERROR << "[" << section << "] get config "
                   << kConfigModelPath << " failed";
    return false;
  }
  LOG_INFO << "[" << section << "] "
                << kConfigModelPath << ":" << model_path_;

  // get model version and latest model path
  // model version is set in model data version file
  // if version file not exist, latested model path is model path
  version_file_path_ = model_path_ + "/" + kVersionFile;
  std::ifstream version_file_handler(version_file_path_.c_str());
  if (!version_file_handler) {
    LOG_WARN << "[" << section << "] open "
                  << version_file_path_ << " failed, "
                  << "model will not be updated";
    version_file_path_ = "";
    latest_model_path_ = model_path_;
  } else {
    std::string line;
    size_t pos = 0;
    while (std::getline(version_file_handler, line)) {
      util::StrUtil::Trim(line);
      // format: model_version=20180816
      if ((pos = line.find(kModelVersionTag)) != std::string::npos) {
        model_version_ = line.substr(pos + kModelVersionTag.length());
        util::StrUtil::Trim(model_version_);
      }
    }

    if (model_version_ == "") {
      LOG_ERROR << "[" << section << "] model_version_str is empty";
      return false;
    }
    LOG_INFO << "[" << section << "] model_version:" << model_version_;

    version_file_handler.close();

    latest_model_path_ = model_path_ + "/" + model_version_;
  }

  LOG_INFO << "[" << section << "] model_data_path: "
                << latest_model_path_;

  return true;
}

}  // namespace tdm_serving
