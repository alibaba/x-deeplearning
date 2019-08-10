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


#pragma once

#include <functional>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "xdl/data_io/op/feature_op/expr/dsl_unit.h"

namespace xdl {
namespace io {

struct ExprNode;

class DslParser;
class ExprGraph;
class Feature;
class FeatureLine;
class FeatureValue;
class SingleFeature;
class MultiFeature;

using FeatureNameVec = std::vector<std::string>;
using FeatureNameMap = std::unordered_map<std::string, int>;

using FeatureMap = std::unordered_map<std::string, Feature *>;
using DslUnitMap = std::unordered_map<std::string, DslUnit>;

using TransformKeyFunc = std::function<int64_t(int64_t key)>;
using TransformValueFunc = std::function<float(float value)>;
using StatisValueFunc = std::function<int(float value0, float value1, float &value)>;

using CombineKeyFunc = std::function<int64_t(int64_t key0, int64_t key1)>;
using CombineValueFunc = std::function<float(float value0, float value1)>;
using FeatureValueCacheMap = std::unordered_map<int64_t, const FeatureValue *>;
using FeatureCacheMap = std::unordered_map<const Feature *, FeatureValueCacheMap>;

}  // namespace io
}  // namespace xdl