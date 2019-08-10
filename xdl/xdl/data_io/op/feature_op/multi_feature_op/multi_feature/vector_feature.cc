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


#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature/vector_feature.h"

#include <xdl/core/utils/logging.h>
#include <math.h>

#include "xdl/data_io/op/feature_op/expr/expr_node.h"
#include "xdl/data_io/op/feature_op/feature_op_constant.h"

namespace xdl {
namespace io {

bool VectorFeature::Combine(std::vector<const ExprNode *> &source_nodes, ExprNode *result_node) {
  XDL_CHECK(source_nodes.size() == 2);
  const int values_size_0 = source_nodes[0]->values_size();
  const int values_size_1 = source_nodes[1]->values_size();
  if (values_size_0 == 0 || values_size_1 == 0)  return false;
  XDL_CHECK(values_size_0 == values_size_1);
  float aa = 0.F, bb = 0.F, ab = 0.F;
  int64_t source_key;
  float source_value_0, source_value_1;
  for (int k = 0; k < values_size_0; ++k) {
    source_nodes[0]->get(k, source_key, source_value_0);
    source_nodes[1]->get(k, source_key, source_value_1);
    aa += combine_value_func_(source_value_0, source_value_0);
    bb += combine_value_func_(source_value_1, source_value_1);
    ab += combine_value_func_(source_value_0, source_value_1);
  }
  result_node->add(kDefaultKey, ab / (sqrtf(aa) + sqrtf(bb) + kTraceFloat));
  return true;
}

}  // namespace io
}  // namespace xdl
