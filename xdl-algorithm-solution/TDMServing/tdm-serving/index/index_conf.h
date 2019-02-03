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

#ifndef TDM_SERVING_INDEX_INDEX_CONF_H_
#define TDM_SERVING_INDEX_INDEX_CONF_H_

#include <string>
#include "util/conf_parser.h"

namespace tdm_serving {

// Index config parsed from config parser
class IndexConf {
 public:
  IndexConf();
  virtual ~IndexConf();

  // Initialize the index config by config parser,
  // section specifies the section name of index in config
  virtual bool Init(const std::string& section,
                    const util::ConfParser& conf_parser);

  void set_section(const std::string& section) {
    section_ = section;
  }

  const std::string& section() const {
    return section_;
  }

  void set_index_path(const std::string& index_path) {
    index_path_ = index_path;
  }

  const std::string& index_path() const {
    return index_path_;
  }

  void set_latest_index_path(const std::string& latest_index_path) {
    latest_index_path_ = latest_index_path;
  }

  const std::string& latest_index_path() const {
    return latest_index_path_;
  }

  void set_version_file_path(const std::string& version_file_path) {
    version_file_path_ = version_file_path;
  }

  const std::string& version_file_path() const {
    return version_file_path_;
  }

  void set_index_version(const std::string& index_version) {
    index_version_ = index_version;
  }

  const std::string& index_version() const {
    return index_version_;
  }

  void set_model_name(const std::string& model_name) {
    model_name_ = model_name;
  }

  const std::string& model_name() const {
    return model_name_;
  }

  void set_model_version(const std::string& model_version) {
    model_version_ = model_version;
  }

  const std::string& model_version() const {
    return model_version_;
  }

  void set_build_omp(const uint32_t build_omp) {
    build_omp_ = build_omp;
  }

  uint32_t build_omp() const {
    return build_omp_;
  }

  void set_filter_name(const std::string& filter_name) {
    filter_name_ = filter_name;
  }

  const std::string& filter_name() const {
    return filter_name_;
  }

 private:
  // section name of index specified in config file
  std::string section_;

  // path of index data file
  std::string index_path_;

  // file path of index with latest version
  // parsed by index version file
  std::string latest_index_path_;

  // path of index version file
  // if file does not exist, new index will not be monitored and reloaded
  std::string version_file_path_;

  // parsed index version specified in version file
  std::string index_version_;

  // specifies the model used by index
  std::string model_name_;

  // version of model
  std::string model_version_;

  // openmp thread num for building index 
  uint32_t build_omp_;

  // specifies the filter used by index
  std::string filter_name_;
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_INDEX_CONF_H_
