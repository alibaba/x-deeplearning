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

// Copyright 2018 Alibaba Inc. All Rights Reserved.

#ifndef TDM_TDM_OP_H_
#define TDM_TDM_OP_H_

#include <typeinfo>

#include <vector>
#include <string>
#include <map>

#include "xdl/data_io/op/op.h"

#include "tdm/dist_tree.h"
#include "tdm/selector.h"
#include "tdm/tree.pb.h"
#include "tdm/common.h"

namespace tdm {

class TDMOP: public xdl::io::Operator {
 public:
  TDMOP();

  bool Init(const std::map<std::string, std::string> &params) override;

  virtual bool Run(xdl::io::SampleGroup *sg);

  bool TDMExpandSample(xdl::io::SampleGroup *sg);

  void GetTargetFeatureIds(xdl::io::SampleGroup *sg,
                           const std::string& target_feature_name,
                           int feature_table_position,
                           std::vector<int64_t>* feature_ids);

  xdl::io::FeatureTable* InsertNewFeatureTable(xdl::io::SampleGroup* sg,
                                               int index);

  std::vector<std::string> split(std::string str, std::string pattern);

 private:
  tdm::Selector* selector_;
  DistTree* tree_;
  int layer_counts_sum_;
  std::vector<int> layer_counts_;
  std::string unit_id_fn_;
  std::vector<int64_t> level_sample_sum_;
  bool expected_count_;
};

}  // namespace tdm

#endif  // TDM_TDM_OP_H_
