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

#ifndef TDM_SERVING_MODEL_BLAZE_BLAZE_PREDICT_CONTEXT_H_
#define TDM_SERVING_MODEL_BLAZE_BLAZE_PREDICT_CONTEXT_H_

#include "model/predict_context.h"
#include "blaze/include/predictor.h"

namespace tdm_serving {

class BlazePredictContext : public PredictContext {
 public:
  BlazePredictContext()
    : predictor_(NULL) {}

  virtual ~BlazePredictContext() {
    delete predictor_;
  }

  void set_predictor(blaze::Predictor* predictor) {
    predictor_ = predictor;
  }

  blaze::Predictor* predictor() const {
    return predictor_;
  }

 private:
  // blaze object, stores session data and predicts score
  blaze::Predictor* predictor_;
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_MODEL_BLAZE_BLAZE_PREDICT_CONTEXT_H_
