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

#ifndef TEST_MODEL_MOCK_MODEL_H_
#define TEST_MODEL_MOCK_MODEL_H_

#include <fstream>
#include "model/model.h"

namespace tdm_serving {

class ModelConf;

class MockModel : public Model {
 public:
  MockModel();
  virtual ~MockModel();

  virtual bool Init(const ModelConf* model_conf);

  virtual bool Predict(PredictContext* predict_ctx,
                       const PredictRequest& predict_req,
                       PredictResponse* predict_res) const;

  virtual PredictContext* GetPredictContext();

  virtual void ReleasePredictContext(PredictContext* context);

 protected:
  virtual ModelConf* CreateModelConf();

 private:
  const ModelConf* mock_model_conf_;

  DISALLOW_COPY_AND_ASSIGN(MockModel);
};

}  // namespace tdm_serving

#endif  // TEST_MODEL_MOCK_MODEL_H_
