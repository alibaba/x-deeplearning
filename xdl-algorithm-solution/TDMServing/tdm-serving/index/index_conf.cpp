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

#include "index/index_conf.h"
#include <algorithm>
#include <fstream>
#include "omp.h"
#include "common/common_def.h"
#include "util/str_util.h"
#include "util/log.h"

namespace tdm_serving {

IndexConf::IndexConf()
  : build_omp_(1) {
}

IndexConf::~IndexConf() {
}

bool IndexConf::Init(const std::string& section,
                     const util::ConfParser& conf_parser) {
  section_ = section;

  // index path
  if (!conf_parser.GetValue<std::string>(section, kConfigIndexPath,
                                         &index_path_)
      || index_path_.empty()) {
    LOG_ERROR << "[" << section << "] get config "
                   << kConfigIndexPath << " failed";
    return false;
  }
  LOG_INFO << "[" << section << "] "
                << kConfigIndexPath << ": " << index_path_;

  // model name
  conf_parser.GetValue<std::string>(section, kConfigIndexModelName,
                                    &model_name_);
  LOG_INFO << "[" << section << "] "
                << kConfigIndexModelName << ":" << model_name_;

  // filter name
  conf_parser.GetValue<std::string>(section, kConfigIndexFilterName,
                                    &filter_name_);
  LOG_INFO << "[" << section << "] "
                << kConfigIndexFilterName << ":" << filter_name_;

  // build omp thread num
  conf_parser.GetValue<uint32_t>(section, kConfigIndexBuildOmp,
                                 1, &build_omp_);
  LOG_INFO << "[" << section << "] build_omp:" << build_omp_;

  // get index version and model version,
  // which are set in version file
  // if version file not exist, latested index path is index path
  version_file_path_ = index_path_ + "/" + kVersionFile;
  std::ifstream version_file_handler(version_file_path_.c_str());
  if (!version_file_handler) {
    LOG_WARN << "[" << section << "] open "
                  << version_file_path_ << " failed, "
                  << "index will not be updated";
    version_file_path_ = "";
    latest_index_path_ = index_path_;
  } else {
    std::string line;
    size_t pos = 0;

    while (std::getline(version_file_handler, line)) {
      util::StrUtil::Trim(line);
      // format: index_version=20180816
      //         model_version=20180818
      if ((pos = line.find(kIndexVersionTag)) != std::string::npos) {
        index_version_ = line.substr(pos + kIndexVersionTag.length());
        util::StrUtil::Trim(index_version_);
      } else if ((pos = line.find(kModelVersionTag)) != std::string::npos) {
        model_version_ = line.substr(pos + kModelVersionTag.length());
        util::StrUtil::Trim(model_version_);
      }
    }

    if (index_version_ == "") {
      LOG_ERROR << "[" << section << "] index_version_str is empty";
      return false;
    }
    LOG_INFO << "[" << section << "] index_version:" << index_version_;
    LOG_INFO << "[" << section << "] model_version:" << model_version_;

    version_file_handler.close();

    latest_index_path_ = index_path_ + "/" + index_version_;
  }

  LOG_INFO << "[" << section << "] index_data_path: "
                << latest_index_path_;

  return true;
}

}  // namespace tdm_serving
