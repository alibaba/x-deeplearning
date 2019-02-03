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

#ifndef TDM_SERVING_MODEL_MODEL_CONF_H_
#define TDM_SERVING_MODEL_MODEL_CONF_H_

#include <string>
#include "common/common_def.h"
#include "util/conf_parser.h"

namespace tdm_serving {

class ModelConf {
 public:
  ModelConf();
  virtual ~ModelConf();

  virtual bool Init(const std::string& section,
                    const util::ConfParser& conf_parser);

  void set_section(const std::string& section) {
    section_ = section;
  }

  const std::string& section() const {
    return section_;
  }

  void set_model_path(const std::string& model_path) {
    model_path_ = model_path;
  }

  const std::string& model_path() const {
    return model_path_;
  }

  void set_latest_model_path(const std::string& latest_model_path) {
    latest_model_path_ = latest_model_path;
  }

  const std::string& latest_model_path() const {
    return latest_model_path_;
  }

  void set_version_file_path(const std::string& version_file_path) {
    version_file_path_ = version_file_path;
  }

  const std::string& version_file_path() const {
    return version_file_path_;
  }

  void set_model_version(const std::string& model_version) {
    model_version_ = model_version;
  }

  const std::string& model_version() const {
    return model_version_;
  }

 private:
  // section name of model specified in config file
  std::string section_;

  // path of model data file
  std::string model_path_;

  // file path of model with latest version
  // parsed by model version file
  std::string latest_model_path_;

  // path of model version file
  // if file does not exist, new model will not be monitored and reloaded
  std::string version_file_path_;

  // parsed model version specified in version file
  std::string model_version_;
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_MODEL_MODEL_CONF_H_
