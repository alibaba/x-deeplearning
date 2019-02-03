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

#ifndef TDM_SERVING_MODEL_MODEL_UNIT_H_
#define TDM_SERVING_MODEL_MODEL_UNIT_H_

#include <string>
#include <vector>
#include "common/common_def.h"
#include "util/conf_parser.h"

namespace tdm_serving {

class Model;

// Manages model reloading
// Uses multi-bufferd model for single-write/multi-read
class ModelUnit {
 public:
  ModelUnit();
  ~ModelUnit();

  bool Init(const std::string& section, const std::string& conf_path);

  bool Reload();

  Model* GetModel(const std::string& model_version = "");

  bool is_enabled();

 private:
  bool enable_;
  std::string model_type_;
  uint32_t idx_;
  std::vector<Model*> model_datas_;
  std::string version_file_path_;

  // config
  std::string section_;
  std::string conf_path_;
  util::ConfParser conf_parser_;

  DISALLOW_COPY_AND_ASSIGN(ModelUnit);
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_MODEL_MODEL_UNIT_H_
