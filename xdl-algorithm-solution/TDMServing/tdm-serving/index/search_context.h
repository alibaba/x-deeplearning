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

#ifndef TDM_SERVING_INDEX_SEARCH_CONTEXT_H_
#define TDM_SERVING_INDEX_SEARCH_CONTEXT_H_

#include <tr1/unordered_set>
#include "common/common_def.h"
#include "model/model_manager.h"
#include "model/predict_context.h"
#include "util/log.h"
#include "proto/search.pb.h"

namespace tdm_serving {

class Filter;

// session data used for searching
class SearchContext {
 public:
  SearchContext()
    : predict_ctx_(NULL), filter_(NULL) {}

  virtual ~SearchContext() {}

  virtual void Clear() {
    // release predict context
    if (predict_ctx_ != NULL) {
      ModelManager::Instance().ReleasePredictContext(model_name_,
                                                     predict_ctx_);
      predict_ctx_ = NULL;
    }
    model_name_.clear();
    filter_ = NULL;
  }

  void set_model_name(const std::string& model_name) {
    model_name_ = model_name;
  }

  const std::string& model_name() const {
    return model_name_;
  }

  void set_filter(Filter* filter) {
    filter_ = filter;
  }

  Filter* filter() {
    return filter_;
  }

  PredictContext* mutable_predict_context() {
    // get predict context by model name
    if (!model_name_.empty() && predict_ctx_ == NULL) {
      predict_ctx_ = ModelManager::Instance().GetPredictContext(model_name_);
    }
    return predict_ctx_;
  }

 private:
  // used to get predict context
  std::string model_name_;

  // session data used for model processing
  PredictContext* predict_ctx_;

  // user defined filter
  Filter* filter_;

  DISALLOW_COPY_AND_ASSIGN(SearchContext);
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_SEARCH_CONTEXT_H_
