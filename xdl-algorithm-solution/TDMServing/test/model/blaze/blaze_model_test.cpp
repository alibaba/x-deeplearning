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

#include <stdio.h>
#include <string.h>
#include <map>
#include "gtest/gtest.h"

#define protected public
#define private public

#include "model/blaze/blaze_def.h"
#include "model/blaze/blaze_model.h"
#include "model/blaze/blaze_predict_context.h"
#include "model/blaze/blaze_model_conf.h"
#include "model/predict_interface.h"
#include "util/log.h"

namespace tdm_serving {

class MockItemFeature : public ItemFeature {
 public:
  MockItemFeature() : feature_group_id_("fg2") {}
  virtual size_t feature_group_size() const {
    return 1;
  }

  virtual const std::string& feature_group_id(size_t /*grp_index*/) const {
    return feature_group_id_;
  }

  virtual size_t feature_entity_size(size_t /*grp_index*/) const {
    return 1;
  }

  virtual uint64_t feature_entity_id(size_t /*grp_index*/,
                                     size_t /*ent_index*/) const {
    return feature_entity_id_;
  }

  virtual float feature_entity_value(size_t /*grp_index*/,
                                     size_t /*ent_index*/) const {
    return feature_entity_value_;
  }

  void set_feature_group_id(const std::string& feature_group_id) {
    feature_group_id_ = feature_group_id;
  }

  void set_feature_entity_id(uint64_t feature_entity_id) {
    feature_entity_id_ = feature_entity_id;
  }

  void set_feature_entity_value(float feature_entity_value) {
    feature_entity_value_ = feature_entity_value;
  } 
 private:
  std::string feature_group_id_;
  uint64_t feature_entity_id_;
  float feature_entity_value_;
};

std::map<std::string, size_t> feed_info;

class MockBlazeModel : public BlazeModel {
protected:
  virtual bool FeedSparseFeatureUInt32(
                    blaze::Predictor* /*predictor*/,
                    const std::string& tensor_name,
                    std::vector<uint32_t>& feature) const {
    LOG_INFO << "request tensor name " << tensor_name;
    LOG_INFO << "request tensor size " << feature.size();
    std::string vs;
    for (size_t j = 0; j < feature.size(); j++) {
      vs = vs + ' ' + std::to_string(feature[j]);
    }
    LOG_INFO << "request tensor content: " << vs;

    feed_info[tensor_name] = feature.size();
    return true;
  }

  virtual bool FeedSparseFeatureUInt64(
                    blaze::Predictor* /*predictor*/,
                    const std::string& tensor_name,
                    std::vector<uint64_t>& feature) const {
    LOG_INFO << "request tensor name " << tensor_name;
    LOG_INFO << "request tensor size " << feature.size();
    std::string vs;
    for (size_t j = 0; j < feature.size(); j++) {
      vs = vs + ' ' + std::to_string(feature[j]);
    }
    LOG_INFO << "request tensor content: " << vs;

    feed_info[tensor_name] = feature.size();
    return true;
  }

  virtual bool FeedSparseFeatureFloat(
                    blaze::Predictor* /*predictor*/,
                    const std::string& tensor_name,
                    std::vector<float>& feature) const {
    LOG_INFO << "request tensor name " << tensor_name;
    LOG_INFO << "request tensor size " << feature.size();
    std::string vs;
    for (size_t j = 0; j < feature.size(); j++) {
      vs = vs + ' ' + std::to_string(feature[j]);
    }
    LOG_INFO << "request tensor content: " << vs;

    feed_info[tensor_name] = feature.size();
    return true;
  }
};

TEST(BlazeModel, set_request) {
  MockBlazeModel* model = new MockBlazeModel();

  BlazeModelConf conf;
  conf.device_type_ = blaze::kPDT_CPU;
  model->blaze_model_conf_ = &conf;

  for (size_t j = 1; j < 12; j++) {
    model->user_tensor_map_["item_" + std::to_string(j)];
  }
  model->ad_tensor_map_["unit_id_expand"];

  // make tdm request
  // user
  UserInfo user_info;
  FeatureGroup* feature_group = NULL;
  FeatureEntity* feature_entity = NULL;

  for (size_t j = 1; j < 10; j++) {
    feature_group = user_info.mutable_user_feature()->add_feature_group();
    feature_group->set_feature_group_id("item_" + std::to_string(j));
    feature_entity = feature_group->add_feature_entity();
    feature_entity->set_id(2 * j);
    feature_entity->set_value(2 * j + 0.1);
    feature_entity = feature_group->add_feature_entity();
    feature_entity->set_id(2 * j + 1);
    feature_entity->set_value(2 * j + 1.1);
  }

  // item
  std::vector<ItemFeature*> item_features;
  MockItemFeature ife1;
  ife1.set_feature_group_id("unit_id_expand");
  ife1.set_feature_entity_id(100);
  ife1.set_feature_entity_value(100.1);
  item_features.push_back(&ife1);
  MockItemFeature ife2;
  ife2.set_feature_group_id("unit_id_expand");
  ife2.set_feature_entity_id(101);
  ife2.set_feature_entity_value(101.1);
  item_features.push_back(&ife2);

  PredictRequest req;
  req.set_user_info(&user_info);
  req.set_item_features(&item_features);

  BlazePredictContext* ctx =
      static_cast<BlazePredictContext*>(model->GetPredictContext());

  feed_info.clear();

  // make request for blaze
  ASSERT_TRUE(model->SetRequest(ctx, req));

  EXPECT_EQ(2, feed_info["item_9.ids"]);
  EXPECT_EQ(2, feed_info["item_9.values"]);
  EXPECT_EQ(1, feed_info["item_9.segments"]);
  EXPECT_EQ(0, feed_info["item_11.ids"]);
  EXPECT_EQ(0, feed_info["item_11.values"]);
  EXPECT_EQ(1, feed_info["item_11.segments"]);
  EXPECT_EQ(2, feed_info["unit_id_expand.ids"]);
  EXPECT_EQ(2, feed_info["unit_id_expand.values"]);
  EXPECT_EQ(2, feed_info["unit_id_expand.segments"]);
}

}  // namespace tdm_serving
