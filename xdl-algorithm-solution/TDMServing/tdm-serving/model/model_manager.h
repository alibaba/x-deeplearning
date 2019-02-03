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

#ifndef TDM_SERVING_MODEL_MODEL_MANAGER_H_
#define TDM_SERVING_MODEL_MODEL_MANAGER_H_

#include <string>
#include "common/common_def.h"
#include "util/conf_parser.h"
#include "util/singleton.h"
#include "util/concurrency/mutex.h"

namespace tdm_serving {

class ModelUnit;
class Model;
class PredictContext;
class PredictRequest;
class PredictResponse;

// Model Manager manages all model instances,
// monitor model version file and reload model.
// It also provide predict interface for all models
class ModelManager : public util::Singleton<ModelManager> {
 public:
  ModelManager();
  ~ModelManager();

  // Initialize by model conf file,
  bool Init(const std::string& conf_path);

  void Reset();

  // Predict interface
  bool Predict(PredictContext* predict_ctx,
               const PredictRequest& predict_req,
               PredictResponse* predict_res);

  // Get sesssion data used for predicting
  PredictContext* GetPredictContext(const std::string& model_name);

  // Release predict context
  void ReleasePredictContext(const std::string& model_name,
                             PredictContext* context);

  // Get model unit by model name
  ModelUnit* GetModelUnit(const std::string& model_name);

  // Check if model exsit
  bool HasModel(const std::string& model_name,
                const std::string& model_version = "");

  // Get model by name and version
  // if version is set to empty, get the latest version
  Model* GetModel(const std::string& model_name,
                  const std::string& model_version = "");

 private:
  typedef std::map<std::string, ModelUnit*> ModelMap;
  ModelMap model_map_;

  util::SimpleMutex mutex_;
  bool inited_;

  DISALLOW_COPY_AND_ASSIGN(ModelManager);
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_MODEL_MODEL_MANAGER_H_
