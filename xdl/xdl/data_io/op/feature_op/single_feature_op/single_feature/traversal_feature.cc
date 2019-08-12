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


#include "xdl/data_io/op/feature_op/single_feature_op/single_feature/traversal_feature.h"

#include "xdl/data_io/op/feature_op/expr/expr_node.h"

namespace xdl {
namespace io {

bool TraversalFeature::Transform(const ExprNode *source_node, ExprNode *result_node) {
  if (result_node->capacity() < source_node->capacity()) {
    result_node->reserve(source_node->capacity());
  }
  int64_t key;
  float value;
  const int size = source_node->values_size();
  for (int i = 0; i < size; ++i) {
    source_node->get(i, key, value);
    result_node->add(transform_key_func_(key), transform_value_func_(value));
  }
  return true;
}

}  // namespace io
}  // namespace xdl