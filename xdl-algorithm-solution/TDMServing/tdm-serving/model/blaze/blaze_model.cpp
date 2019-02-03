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

#include <set>

#include "common/common_def.h"
#include "model/blaze/blaze_model.h"
#include "model/blaze/blaze_def.h"
#include "model/blaze/blaze_model_conf.h"
#include "model/blaze/blaze_predict_context.h"
#include "model/predict_interface.h"
#include "util/str_util.h"
#include "util/log.h"
#include "util/object_free_list.h"

namespace tdm_serving {

BlazeModel::BlazeModel() : obj_pool_(ktObjectPoolInitSize),
    need_to_feed_indicator_(false) {
}

BlazeModel::~BlazeModel() {
}

bool BlazeModel::Init(const ModelConf* model_conf) {
  blaze_model_conf_ = static_cast<const BlazeModelConf*>(model_conf);

  const std::string& section = blaze_model_conf_->section();

  if (!Model::Init(model_conf)) {
    LOG_ERROR << "[" << section << "] init upper model failed";
    return false;
  }

  // load sparse model weight
  std::string sparse_model_weight_file_path =
      blaze_model_conf_->latest_model_path() +
          '/' +kBlazeSparseModelWeightFileName;

  if (!predictor_manager_.LoadSparseModelWeight(
              sparse_model_weight_file_path.c_str())) {
    LOG_ERROR << "[" << section <<"] "
            "blaze load sparse model weight failed";
    return false;
  }
  LOG_DEBUG << "[" << section <<"] "
          "blaze load sparse model weight succ, "
          "file: " << sparse_model_weight_file_path;

  // load model
  std::string model_file_path =
      blaze_model_conf_->latest_model_path() + '/' + kBlazeModelFileName;

  if (!predictor_manager_.LoadModel(model_file_path.c_str())) {
    LOG_ERROR << "[" << section << "] blaze load model failed";
    return false;
  }
  LOG_DEBUG << "[" << section << "] blaze load model succ "
                 << "file: " << model_file_path;

  predictor_manager_.SetRunMode("hybrid");

  // get feature group name to feed
  blaze::Predictor* predictor = predictor_manager_.CreatePredictor(
      blaze_model_conf_->device_type());
  if (predictor == NULL) {
    LOG_ERROR << "[" << section << "] blaze can not get predictor";
    return false;
  }
  const std::vector<std::string>& feed_name_list =
          predictor->ListInputName();
  for (size_t i = 0; i < feed_name_list.size(); i++) {
    blaze::FeedNameConfig feed_name_config
            = predictor->GetFeedNameConfig(feed_name_list[i]);
    if (feed_name_list[i] ==
        blaze::FeedNameUtility::IndicatorLevel2FeedName(0)) {
      need_to_feed_indicator_ = true;
      continue;
    }
    if (feed_name_config.level == 0) {
      ad_tensor_map_[feed_name_config.feature_name];
    } else if (feed_name_config.level == 1) {
      user_tensor_map_[feed_name_config.feature_name];
    }
  }
  delete predictor;

  // pre-allloc blaze predictors
  std::vector<PredictContext*> predictors;
  for (size_t i = 0; i < kPreAllocPredictorNum; i++) {
    predictors.push_back(GetPredictContext());
  }
  for (size_t i = 0; i < predictors.size(); i++) {
    ReleasePredictContext(predictors[i]);
  }

  return true;
}

ModelConf* BlazeModel::CreateModelConf() {
  return new BlazeModelConf();
}

bool BlazeModel::Predict(PredictContext* predict_ctx,
                         const PredictRequest& predict_req,
                         PredictResponse* predict_res) const {
  BlazePredictContext* ctx = static_cast<BlazePredictContext*>(predict_ctx);
  blaze::Predictor* predictor = ctx->predictor();

  // set request
  if (!SetRequest(ctx, predict_req)) {
    return false;
  }

  // predict
  if (!predictor->Forward()) {
    return false;
  }

  if (!ParseResponse(ctx, predict_req, predict_res)) {
    return false;
  }

  return true;
}

PredictContext* BlazeModel::GetPredictContext() {
  BlazePredictContext* ctx = obj_pool_.Acquire();
  if (ctx->predictor() == NULL) {
    blaze::Predictor* predictor = predictor_manager_.CreatePredictor(
        blaze_model_conf_->device_type());
    ctx->set_predictor(predictor);
  }
  return ctx;
}

void BlazeModel::ReleasePredictContext(PredictContext* context) {
  obj_pool_.Release(static_cast<BlazePredictContext*>(context),
                    false);  // false, no erase obj
}

bool BlazeModel::SetRequest(BlazePredictContext* ctx,
                            const PredictRequest& predict_req) const {
  blaze::Predictor* predictor = ctx->predictor();

  // indicator
  if (need_to_feed_indicator_ == true) {
    size_t ad_num = predict_req.item_features()->size();
    std::vector<uint32_t> indicators(ad_num, 0);

    std::string indicator_tensor_name =
        blaze::FeedNameUtility::IndicatorLevel2FeedName(0);
    if (!FeedSparseFeatureUInt32(predictor, indicator_tensor_name,
                                 indicators)) {
      return false;
    }
  }

  // user feature
  std::map<std::string, TensorInfo> user_tensor_map = user_tensor_map_;

  const FeatureGroupList& user_feature =
          predict_req.user_info()->user_feature();
  for (int i = 0; i < user_feature.feature_group_size(); i++) {
    const FeatureGroup& feature_group = user_feature.feature_group(i);
    const std::string& feature_group_id = feature_group.feature_group_id();

    auto it = user_tensor_map.find(feature_group_id);
    if (it == user_tensor_map.end()) {
      continue;
    }
    TensorInfo& ti = it->second;

    for (int j = 0; j < feature_group.feature_entity_size(); j++) {
      const FeatureEntity& feature_entity = feature_group.feature_entity(j);
      ti.ids.push_back(feature_entity.id());
      ti.values.push_back(feature_entity.value());
    }
    ti.segs.push_back(feature_group.feature_entity_size());
  }

  for (auto &ti : user_tensor_map) {
    std::string id_tensor_name =
        blaze::FeedNameUtility::SparseFeatureName2FeedName(
            ti.first, blaze::kSparseFeatureId);
    if (!FeedSparseFeatureUInt64(predictor, id_tensor_name,
                                 ti.second.ids)) {
      return false;
    }

    std::string value_tensor_name =
        blaze::FeedNameUtility::SparseFeatureName2FeedName(
            ti.first, blaze::kSparseFeatureValue);
    if (!FeedSparseFeatureFloat(predictor, value_tensor_name,
                                ti.second.values)) {
      return false;
    }

    if (ti.second.ids.size() == 0) {
      ti.second.segs.push_back(0);
    }

    std::string seg_tensor_name =
        blaze::FeedNameUtility::SparseFeatureName2FeedName(
            ti.first, blaze::kAuxSparseFeatureSegment);
    if (!FeedSparseFeatureUInt32(predictor, seg_tensor_name,
                                 ti.second.segs)) {
      return false;
    }
  }

  // ad feature
  std::map<std::string, TensorInfo> ad_tensor_map = ad_tensor_map_;
  std::set<std::string> has_filled;

  const std::vector<ItemFeature*>* item_features = predict_req.item_features();
  for (size_t i = 0; i < item_features->size(); i++) {
    const ItemFeature* item_feature = item_features->at(i);
    has_filled.clear();

    for (size_t j = 0; j < item_feature->feature_group_size(); j++) {
      const std::string& feature_group_id = item_feature->feature_group_id(j);

      auto it = ad_tensor_map.find(feature_group_id);
      if (it == ad_tensor_map.end()) {
        continue;
      }
      TensorInfo& ti = it->second;

      for (size_t k = 0; k < item_feature->feature_entity_size(j); k++) {
        ti.ids.push_back(item_feature->feature_entity_id(j, k));
        ti.values.push_back(item_feature->feature_entity_value(j, k));
      }
      ti.segs.push_back(item_feature->feature_entity_size(j));
      has_filled.insert(feature_group_id);
    }

    for (auto &ti : ad_tensor_map) {
      if (has_filled.count(ti.first) == 0) {
        ti.second.segs.push_back(0);
      }
    }
  }

  for (auto &ti : ad_tensor_map) {
    std::string id_tensor_name =
        blaze::FeedNameUtility::SparseFeatureName2FeedName(
            ti.first, blaze::kSparseFeatureId);
    if (!FeedSparseFeatureUInt64(predictor, id_tensor_name,
                                 ti.second.ids)) {
      return false;
    }

    std::string value_tensor_name =
        blaze::FeedNameUtility::SparseFeatureName2FeedName(
            ti.first, blaze::kSparseFeatureValue);
    if (!FeedSparseFeatureFloat(predictor, value_tensor_name,
                                ti.second.values)) {
      return false;
    }

    std::string seg_tensor_name =
        blaze::FeedNameUtility::SparseFeatureName2FeedName(
            ti.first, blaze::kAuxSparseFeatureSegment);
    if (!FeedSparseFeatureUInt32(predictor, seg_tensor_name,
                                 ti.second.segs)) {
      return false;
    }
  }

  return true;

}

bool BlazeModel::ParseResponse(BlazePredictContext* ctx,
                               const PredictRequest& predict_req,
                               PredictResponse* predict_res) const {
  blaze::Predictor* predictor = ctx->predictor();

  if (predictor->OutputSize() != 1) {
    LOG_ERROR << "predictor output size is not 1";
    return false;
  }

  void* output = NULL;
  size_t len = 0;
  size_t idx = 0;

  if (!predictor->Output(idx, &output, &len)) {
    LOG_ERROR << "predictor get output failed";
    return false;
  }

  float* float_output = reinterpret_cast<float*>(output);
  size_t float_len = len / sizeof(float);

#ifndef NDEBUG
  for (size_t i = 0; i < float_len; i++) {
    LOG_DEBUG << "result tensor score" << float_output[i];
  }
#endif

  if (float_len != predict_req.item_features()->size() * 2) {
    LOG_ERROR << "predictor output tensor len " << float_len
                   << "!= item feature size "
                   << predict_req.item_features()->size() << "* 2";
    return false;
  }

  for (size_t i = 0; i < predict_req.item_features()->size(); i++) {
    predict_res->add_score(float_output[i * 2 + 1]);
  }

  return true;
}

bool BlazeModel::FeedSparseFeatureUInt32(
              blaze::Predictor* predictor,
              const std::string& tensor_name,
              std::vector<uint32_t>& feature) const {
  return FeedSparseFeature(predictor, tensor_name, feature);
}

bool BlazeModel::FeedSparseFeatureUInt64(
              blaze::Predictor* predictor,
              const std::string& tensor_name,
              std::vector<uint64_t>& feature) const {
  return FeedSparseFeature(predictor, tensor_name, feature);
}

bool BlazeModel::FeedSparseFeatureFloat(
              blaze::Predictor* predictor,
              const std::string& tensor_name,
              std::vector<float>& feature) const {
  return FeedSparseFeature(predictor, tensor_name, feature);
}

template <class T>
bool BlazeModel::FeedSparseFeature(
            blaze::Predictor* predictor,
            const std::string& tensor_name,
            std::vector<T>& feature) const {
#ifndef NDEBUG
  LOG_DEBUG << "request tensor name " << tensor_name;
  LOG_DEBUG << "request tensor size " << feature.size();
  std::string vs;
  for (size_t j = 0; j < feature.size(); j++) {
    vs = vs + ' ' + std::to_string(feature[j]);
  }
  LOG_DEBUG << "request tensor content: " << vs;
#endif

  if (feature.size() == 0) {
    return true;
  }

  if (!predictor->ReshapeInput(tensor_name.c_str(), { feature.size() })) {
    LOG_ERROR << "reshape predictor input by name "
                   << tensor_name << " failed";
    return false;
  }
  if (!predictor->Feed(tensor_name.c_str(), feature.data(),
                       feature.size() * sizeof(T))) {
    LOG_ERROR << "feed predictor input by name "
                   << tensor_name << " failed";
    return false;
  }
  return true;
}

// register itself
REGISTER_MODEL(blaze, BlazeModel);

}  // namespace tdm_serving
