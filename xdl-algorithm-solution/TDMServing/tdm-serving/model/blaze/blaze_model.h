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

#ifndef TDM_SERVING_MODEL_BLAZE_BLAZE_MODEL_H_
#define TDM_SERVING_MODEL_BLAZE_BLAZE_MODEL_H_

#include "model/model.h"
#include "blaze/include/predictor.h"
#include "util/concurrency/object_pool.h"

namespace tdm_serving {

class ModelConf;
class BlazeModelConf;
class BlazePredictContext;
class PredictRequest;
class PredictResponse;
class ItemFeature;

struct TensorInfo {
  std::vector<uint64_t> ids;
  std::vector<float> values;
  std::vector<uint32_t> segs;
};

class BlazeModel : public Model {
 public:
  BlazeModel();

  virtual ~BlazeModel();

  virtual bool Init(const ModelConf* model_conf);

  virtual bool Predict(PredictContext* predict_ctx,
                       const PredictRequest& predict_req,
                       PredictResponse* predict_res) const;

  virtual PredictContext* GetPredictContext();

  virtual void ReleasePredictContext(PredictContext* context);

 protected:
  virtual ModelConf* CreateModelConf();

  virtual bool FeedSparseFeatureUInt32(
                        blaze::Predictor* predictor,
                        const std::string& tensor_name,
                        std::vector<uint32_t>& feature) const;

  virtual bool FeedSparseFeatureUInt64(
                        blaze::Predictor* predictor,
                        const std::string& tensor_name,
                        std::vector<uint64_t>& feature) const;

  virtual bool FeedSparseFeatureFloat(
                        blaze::Predictor* predictor,
                        const std::string& tensor_name,
                        std::vector<float>& feature) const;

 private:
  bool SetRequest(BlazePredictContext* ctx,
                  const PredictRequest& predict_req) const;

  bool ParseResponse(BlazePredictContext* ctx,
                     const PredictRequest& predict_req,
                     PredictResponse* predict_res) const;

  template <class T>
  bool FeedSparseFeature(blaze::Predictor* predictor,
                         const std::string& tensor_name,
                         std::vector<T>& feature) const;

 private:
  // model conf
  const BlazeModelConf* blaze_model_conf_;

  // blaze preditor
  blaze::PredictorManager predictor_manager_;

  // object pool
  // BlazePredictContext can only be for one blaze model,
  // so we can not use object free list
  util::ObjectPool<BlazePredictContext, util::ObjectPoolNormalAllocator,
                   util::AdaptiveMutex> obj_pool_;

  // need to feed indicator
  bool need_to_feed_indicator_;

  // user feature group name for input tensor
  std::map<std::string, TensorInfo> user_tensor_map_;

  // ad feature group name for input tensor
  std::map<std::string, TensorInfo> ad_tensor_map_;

  DISALLOW_COPY_AND_ASSIGN(BlazeModel);
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_MODEL_BLAZE_BLAZE_MODEL_H_
