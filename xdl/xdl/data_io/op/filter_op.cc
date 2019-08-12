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

#include "xdl/data_io/op/filter_op.h"

#include <iostream>

#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

static const char CTRL_A = '\001';

static std::string GetStag(const std::string &skey) {
  auto pos = strchr(skey.c_str(), CTRL_A);
  if (pos == nullptr) {
    return skey;
  }
  int len = pos - skey.c_str();
  XDL_CHECK(len > 0 && len < 1024) << len;
  return std::string(skey.c_str(), len);
}

bool FilterOP::Init(const std::map<std::string, std::string> &params) {
  for (auto &kv: params) {
    std::cout << "init: " << kv.first << "->" << kv.second << std::endl;
    if (kv.first == "stag") {
      auto &stags = kv.second;
      std::string::size_type end, beg = 0;
      do {
        end = stags.find(',', beg);
        del_skeys_.insert(stags.substr(beg, end-beg));
        beg = end + 1;
      } while (end != std::string::npos);
    } else {
      XDL_LOG(FATAL) << "unkown param: " << kv.first;
    }
  }
  return true;
}

bool FilterOP::Run(SampleGroup *sg) {
  if (del_skeys_.empty()) {
    return true;
  }

  std::vector<bool> dels;

  auto &sample_ids = sg->sample_ids();
  auto &labels = sg->labels();

  unsigned filtered = 0;
  unsigned passed = 0;

  for (int i = 0; i < sample_ids.size(); ++i) {
    auto stag = GetStag(sample_ids.Get(i));
    if (del_skeys_.find(stag) != del_skeys_.end()) {
      dels.push_back(true);
      ++ filtered;
    } else {
      dels.push_back(false);
      ++ passed;
    }
  }

  if (filtered == 0) {
    passed_ += passed;
    return true;
  }

  google::protobuf::RepeatedPtrField<std::string> passed_sample_ids;
  google::protobuf::RepeatedPtrField<Label> passed_labels;
  google::protobuf::RepeatedPtrField<FeatureLine> passed_fls;

  /// rewrite sample ids
  passed_sample_ids.Reserve(passed);
  for (int i = 0; i < sample_ids.size(); ++i) {
    if (dels[i]) {
      continue;
    }
    auto *val = passed_sample_ids.Add();
    *val = sample_ids[i];
  }
  sg->mutable_sample_ids()->CopyFrom(passed_sample_ids);


  /// rewrite labels
  passed_labels.Reserve(passed);
  for (int i = 0; i < labels.size(); ++i) {
    if (dels[i]) {
      continue;
    }
    auto *val = passed_labels.Add();
    //val->CopyFrom(labels.Get(i));
    *val = labels.Get(i);
  }
  sg->mutable_labels()->CopyFrom(passed_labels);

  /// rewrite feature tables
  auto ft = sg->mutable_feature_tables(0);
  auto &fls = ft->feature_lines();
  for (int i = 0; i < fls.size(); ++i) {
    if (dels[i]) {
      continue;
    }
    auto *val = passed_fls.Add();
    //val->CopyFrom(fls.Get(i));
    *val = fls.Get(i);
  }
  ft->mutable_feature_lines()->CopyFrom(passed_fls);

  size_t n = sg->labels_size();
  XDL_CHECK(sg->sample_ids_size() == 0 || sg->sample_ids_size() == n)
      << "sample_id.size=" << sg->sample_ids_size() << " n=" << n;
  XDL_CHECK(sg->feature_tables(0).feature_lines_size() == n)
      << "table[0].size=" << sg->feature_tables(0).feature_lines_size() << " n=" << n;

  passed_ += passed;
  filtered_ += filtered;
  //std::cout << "passed=" << passed_ << " filtered=" << filtered_ << std::endl;

  return true;
}

std::map<std::string, std::string> FilterOP::URun(const std::map<std::string, std::string> &params) {
  std::map<std::string, std::string> ret;
  ret["filtered"] = std::to_string(filtered_);
  ret["passed"] = std::to_string(passed_);
  return ret;
} 

XDL_REGISTER_IOP(FilterOP)

}  // namespace io
}  // namespace xdl
