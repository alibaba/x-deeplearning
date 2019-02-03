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

#include "model/blaze/blaze_model_conf.h"
#include "model/blaze/blaze_def.h"
#include "util/str_util.h"
#include "util/log.h"

namespace tdm_serving {

BlazeModelConf::BlazeModelConf() {
}

BlazeModelConf::~BlazeModelConf() {
}

bool BlazeModelConf::Init(const std::string& section,
                          const util::ConfParser& conf_parser) {
  if (!ModelConf::Init(section, conf_parser)) {
    LOG_ERROR << "[" << section << "] model conf init failed";
    return false;
  }

  // device type
  uint32_t device_type_num = 0;
  conf_parser.GetValue<uint32_t>(section, kConfigBlazeDeviceType,
      static_cast<uint32_t>(blaze::kPDT_CPU), &device_type_num);
  switch (device_type_num) {
    case 0:
      device_type_ = blaze::kPDT_CPU;
      LOG_INFO << "[" << section << "] device_type: CPU";
      break;
    case 1:
      device_type_ = blaze::kPDT_CUDA;
      LOG_INFO << "[" << section << "] device_type: CUDA";
      break;
    default:
      LOG_ERROR << "[" << section << "] unknown device_type: "
                     << device_type_num;
      return false;
  }

  return true;
}

}  // namespace tdm_serving
