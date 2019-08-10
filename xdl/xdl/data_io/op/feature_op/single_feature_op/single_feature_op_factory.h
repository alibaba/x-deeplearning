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

#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_op.h"

namespace xdl {
namespace io {

class SingleFeatureOpFactory {
 public:
  static SingleFeatureOp *Get(SingleFeaOpType single_fea_op_type,
                              TransformKeyFunc transform_key_func,
                              TransformValueFunc transform_value_func,
                              StatisValueFunc statis_value_func,
                              bool is_average = false) {
    SingleFeatureOp *ret = new SingleFeatureOp();
    ret->Init(single_fea_op_type,
              transform_key_func,
              transform_value_func,
              statis_value_func,
              is_average);
    return ret;
  }

  static void Release(SingleFeatureOp *single_feature_op) {
    single_feature_op->Destroy();
    //delete single_feature_op;
  }
};

}  // namespace io
}  // namespace xdl