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

#ifndef TDM_SERVING_MODEL_PREDICT_INTERFACE_H_
#define TDM_SERVING_MODEL_PREDICT_INTERFACE_H_

#include <string>
#include <vector>
#include "common/common_def.h"
#include "proto/search.pb.h"

namespace tdm_serving {

// Item feature interface
class ItemFeature {
 public:
  ItemFeature() {}

  virtual ~ItemFeature() {}

  virtual size_t feature_group_size() const = 0;

  virtual const std::string& feature_group_id(size_t grp_index) const = 0;

  virtual size_t feature_entity_size(size_t grp_index) const = 0;

  virtual uint64_t feature_entity_id(size_t grp_index,
                                     size_t ent_index) const = 0;

  virtual float feature_entity_value(size_t grp_index,
                                     size_t ent_index) const = 0;
};

// Model layer interface, predict request
class PredictRequest {
 public:
  PredictRequest() : user_info_(NULL), item_features_(NULL) {
  }

  ~PredictRequest() {}

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

  const UserInfo* user_info() const {
    return user_info_;
  }

  void set_user_info(const UserInfo* user_info) {
    user_info_ = user_info;
  }

  const std::vector<ItemFeature*>* item_features() const {
    return item_features_;
  }

  void set_item_features(const std::vector<ItemFeature*>* item_features) {
    item_features_ = item_features;
  }

 private:
  // name of model, used for locate model
  std::string model_name_;

  // version of model, used for locate model
  std::string model_version_;

  // user features
  const UserInfo* user_info_;

  // item features
  const std::vector<ItemFeature*>* item_features_;
};

// Model layer interface, predict response
class PredictResponse {
 public:
  PredictResponse() {}
  ~PredictResponse() {}

  size_t score_size() {
    return scores_.size();
  }

  float score(size_t index) {
    return scores_[index];
  }

  void add_score(float score) {
    scores_.push_back(score);
  }

 private:
  // item scores
  std::vector<float> scores_;
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_MODEL_PREDICT_INTERFACE_H_
