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

#include "xdl/data_io/op/feature_op/feature_op_type.h"
#include "xdl/data_io/op/op.h"

namespace xdl {
namespace io {

class FeatureOP : public Operator {
 public:
  virtual ~FeatureOP() = default;

  bool Init(const std::vector<std::string> &dsl_arr,
            const std::vector<FeatureNameVec> &feature_name_vecs);
  virtual bool Init(const std::map<std::string, std::string> &params) override;
  virtual bool Run(SampleGroup *sample_group) override;

 protected:
  void InitFeatureNameStore();
  int BinarySearch(const FeatureLine *feature_line, const std::string &feature_name, int &begin);

 private:
  ExprGraph *expr_graph_ = nullptr;
  FeatureNameMap feature_name_map_;
  std::vector<FeatureNameVec> feature_name_vecs_;
  std::vector<size_t> feature_name_vec_sizes_;
  bool inited_ = false;
};

}  // namespace io
}  // namespace xdl