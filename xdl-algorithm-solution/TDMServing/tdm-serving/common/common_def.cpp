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

#include "common/common_def.h"

namespace tdm_serving {

const std::string kMetaSection = "meta";

const std::string kConfigEnable = "enable";
const std::string kConfigIndexType = "type";
const std::string kConfigIndexPath = "index_path";
const std::string kConfigIndexModelName = "model_name";
const std::string kConfigIndexFilterName = "filter_name";
const std::string kConfigIndexBuildOmp = "build_thread_num";
const std::string kConfigModelType = "type";
const std::string kConfigModelPath = "model_path";
const std::string kConfigFilterType = "type";

const std::string kVersionFile = "version";
const std::string kIndexVersionTag = "index_version=";
const std::string kModelVersionTag = "model_version=";

const uint32_t kIndexInstanceNum = 2;
const uint32_t kModelInstanceNum = 2;

const uint32_t ktObjectPoolInitSize = 50;

}  // namespace tdm_serving

