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

#ifndef TDM_SERVING_MODEL_MODEL_H_
#define TDM_SERVING_MODEL_MODEL_H_

#include <string>
#include "common/common_def.h"
#include "model/model_conf.h"
#include "util/conf_parser.h"
#include "util/registerer.h"

namespace tdm_serving {

class ModelConf;
class PredictContext;
class PredictRequest;
class PredictResponse;

class Model {
 public:
  Model();
  virtual ~Model();

  // Initialize the model by parsed config,
  // section specifies the section name of model in config
  bool Init(const std::string& section,
            const util::ConfParser& conf_parser);

  // Initialize by parsed model config
  virtual bool Init(const ModelConf* model_conf);

  // Do search
  virtual bool Predict(PredictContext* predict_ctx,
                       const PredictRequest& predict_req,
                       PredictResponse* predict_res) const = 0;

  // Get sesssion data used for predicting
  virtual PredictContext* GetPredictContext() = 0;

  // Release predict context
  virtual void ReleasePredictContext(PredictContext* context) = 0;

  // Get paht of version file used for model reloading
  const std::string& version_file_path() {
    return model_conf_->version_file_path();
  }

  const std::string& model_name() {
    return model_conf_->section();
  }

  const std::string& model_version() {
    return model_conf_->model_version();
  }

 protected:
  virtual ModelConf* CreateModelConf();

 private:
  ModelConf* model_conf_;

  DISALLOW_COPY_AND_ASSIGN(Model);
};

// define register
REGISTER_REGISTERER(Model);
#define REGISTER_MODEL(title, name) \
    REGISTER_CLASS(Model, title, name)

}  // namespace tdm_serving

#endif  // TDM_SERVING_MODEL_MODEL_H_
