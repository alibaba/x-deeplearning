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


#include "xdl/data_io/op/feature_op/source_feature_op/source_feature_op.h"

#include <xdl/core/utils/logging.h>

namespace xdl {
namespace io {

bool SourceFeatureOp::Init(const std::string &name) {
  name_ = name;
  return true;
}

bool SourceFeatureOp::Destroy() {
  return true;
}

bool SourceFeatureOp::Run(const FeatureMap *feature_map, void *&result_feature) {
  const auto &iter = feature_map->find(name_);
  if (iter == feature_map->end())  return false;
  result_feature = iter->second;
  XDL_CHECK(result_feature != nullptr);
  return true;
}

}  // namespace io
}  // namespace xdl
