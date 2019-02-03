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

#include "model/model.h"

namespace tdm_serving {

Model::Model()
    : model_conf_(NULL) {
}

Model::~Model() {
  DELETE_AND_SET_NULL(model_conf_);
}

bool Model::Init(const std::string& section,
                 const util::ConfParser& conf_parser) {
  model_conf_ = CreateModelConf();
  model_conf_->Init(section, conf_parser);

  if (!Init(model_conf_)) {
    return false;
  }

  return true;
}

ModelConf* Model::CreateModelConf() {
  return new ModelConf();
}

bool Model::Init(const ModelConf* /*model_conf*/) {
  return true;
}

}  // namespace tdm_serving
