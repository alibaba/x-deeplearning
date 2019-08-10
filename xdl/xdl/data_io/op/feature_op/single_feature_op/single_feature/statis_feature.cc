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


#include "xdl/data_io/op/feature_op/single_feature_op/single_feature/statis_feature.h"

#include <xdl/core/utils/logging.h>

#include "xdl/data_io/op/feature_op/expr/expr_node.h"
#include "xdl/data_io/op/feature_op/feature_op_constant.h"

namespace xdl {
namespace io {

void StatisFeature::Init(TransformKeyFunc transform_key_func,
                         TransformValueFunc transform_value_func,
                         StatisValueFunc statis_value_func,
                         bool is_average) {
  XDL_CHECK(statis_value_func != nullptr);
  SingleFeature::Init(transform_key_func, transform_value_func, statis_value_func, is_average);
}

bool StatisFeature::Transform(const ExprNode *source_node, ExprNode *result_node) {
  if (source_node->values_size() == 0)  return false;
  int64_t key;
  float result = Statis(source_node, key);
  result_node->add(key, result);
  return true;
}

float StatisFeature::Statis(const ExprNode *source_node, int64_t &key) {
  float result = source_node->value(0);
  key = source_node->key(0);
  const int size = source_node->values_size();
  for (int i = 1; i < size; ++i) {
	int ret = statis_value_func_(result, source_node->value(i), result);
	if (ret < 0)  key = kDefaultKey;
	else if (ret > 0)  key = source_node->key(i);
  }
  if (is_average_)  result /= size;
  if (transform_value_func_ != nullptr)  result = transform_value_func_(result);
  return result;
}

}  // namespace io
}  // namespace xdl
