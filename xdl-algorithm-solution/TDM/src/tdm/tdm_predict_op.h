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

#ifndef TDM_TDM_PREDICT_OP_H_
#define TDM_TDM_PREDICT_OP_H_

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <mutex>
#include <typeinfo>
#include <fstream>

#include "xdl/data_io/op/op.h"

#include "tdm/dist_tree.h"
#include "tdm/selector.h"
#include "tdm/tree.pb.h"

namespace tdm {

class TDMPREDICTOP: public xdl::io::Operator {
 public:
  TDMPREDICTOP();

  bool Init(const std::map<std::string, std::string>& params) override;
  std::map<std::string, std::string> URun(
      const std::map<std::string, std::string> &params) override;

  virtual bool Run(xdl::io::SampleGroup* sg);

  bool TDMExpandSample(xdl::io::SampleGroup* sg);
  bool VectorReExpandSample(xdl::io::SampleGroup *sg);
  void GetAllLevelId(std::vector<int64_t> *ids, int level);

  std::vector<int64_t> GetTargetFeatureIds(
      xdl::io::SampleGroup* sg, std::string target_feature_name,
      int feature_table_position);

  std::vector<std::pair<int64_t, float>> GetTargetFeatureIdsWithValue(
      xdl::io::SampleGroup* sg, std::string target_feature_name,
      int feature_table_position);

  bool AddFeatureIds(xdl::io::SampleGroup* sg, int feature_table_index,
                     int feature_line_index, std::string target_id_fn,
                     std::vector<std::pair<int64_t, float> >* add_ids,
                     int default_value);
  void GetProbs(xdl::io::SampleGroup *sg, std::vector<float> * probs,
                int index);

  xdl::io::FeatureTable* InsertNewFeatureTable(xdl::io::SampleGroup *sg,
                int index);

 private:
  DistTree* tree_;
  std::string gt_id_fn_;
  std::string pred_id_fn_;
  std::string unit_id_expand_fn_;
  std::string predict_result_file_;
  std::ofstream predict_result_file_stream_;
  std::string expand_mode_;
  int start_predict_layer_;

  // TopK
  int level_topk_;
  int final_topk_;

  // 全局用户数量
  int global_sample_num_;

  // 总召回数量
  int global_r_num_;

  // 总真集数量
  int global_gt_num_;

  // 总预测数量
  int global_p_num_;

  // 平均召回率sum
  float avg_r_sum_;

  // 平均准确率sum
  float avg_p_sum_;
  std::mutex mutex_;
};

}  // namespace tdm

#endif  // TDM_TDM_PREDICT_OP_H_
