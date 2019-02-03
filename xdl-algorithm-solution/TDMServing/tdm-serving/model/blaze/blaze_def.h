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

#ifndef TDM_SERVING_MODEL_BLAZE_BLAZE_DEF_H_
#define TDM_SERVING_MODEL_BLAZE_BLAZE_DEF_H_

#include <inttypes.h>
#include <string>
#include "common/common_def.h"

namespace tdm_serving {

extern const std::string kConfigBlazeDeviceType;

extern const std::string kBlazeModelFileName;
extern const std::string kBlazeSparseModelWeightFileName;

extern const uint32_t kPreAllocPredictorNum;

}  // namespace tdm_serving

#endif  // TDM_SERVING_MODEL_BLAZE_BLAZE_DEF_H_
