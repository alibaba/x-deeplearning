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

#ifndef TDM_SERVING_MODEL_BLAZE_BLAZE_MODEL_CONF_H_
#define TDM_SERVING_MODEL_BLAZE_BLAZE_MODEL_CONF_H_

#include "model/model_conf.h"
#include "blaze/include/predictor.h"

namespace tdm_serving {

class BlazeModelConf : public ModelConf {
 public:
  BlazeModelConf();
  virtual ~BlazeModelConf();

  virtual bool Init(const std::string& section,
                    const util::ConfParser& conf_parser);

  void set_device_type(blaze::PredictDeviceType device_type) {
    device_type_ = device_type;
  }

  blaze::PredictDeviceType device_type() const {
    return device_type_;
  }

 private:
  // predict device type, CPU or GPU
  blaze::PredictDeviceType device_type_;
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_MODEL_BLAZE_BLAZE_MODEL_CONF_H_
