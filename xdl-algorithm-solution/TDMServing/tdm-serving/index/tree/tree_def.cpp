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

#include "index/tree/tree_def.h"

namespace tdm_serving {

const std::string kConfigTreeLevelTopN = "tree_level_topn";
const std::string kConfigItemFeatureGroupId = "item_feature_group_id";

const uint32_t kDefaultTreeLevelTopN = 512;

const std::string kTreeMetaFileName = "meta.dat";
const std::string kTreeDataFilePrefix = "tree.dat.";

const uint32_t kRootLevel = 0;


}  // namespace tdm_serving
