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

#include "model/blaze/blaze_def.h"

namespace tdm_serving {

const std::string kConfigBlazeDeviceType = "blaze_device_type";

const std::string kBlazeModelFileName = "model.dat";
const std::string kBlazeSparseModelWeightFileName = "sparse_qed";

const uint32_t kPreAllocPredictorNum = 50;

}  // namespace tdm_serving
