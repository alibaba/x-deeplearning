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

#include "index/tree/tree_index_conf.h"
#include <algorithm>
#include "omp.h"
#include "index/tree/tree_def.h"
#include "util/str_util.h"
#include "util/log.h"

namespace tdm_serving {

TreeIndexConf::TreeIndexConf()
  : model_batch_num_(1) {
}

TreeIndexConf::~TreeIndexConf() {
}

bool TreeIndexConf::Init(const std::string& section,
                         const util::ConfParser& conf_parser) {
  // upper init
  if (!IndexConf::Init(section, conf_parser)) {
    LOG_ERROR << "[" << section << "] upper Index::Init failed";
    return false;
  }

  // tree_level_topn
  std::string level_topn_conf_str;
  if (!conf_parser.GetValue<std::string>(
      section, kConfigTreeLevelTopN, &level_topn_conf_str)) {
    LOG_ERROR << "[" << section << "] get config "
                   << kConfigTreeLevelTopN << " failed";
    return false;
  }
  if (!ParseTreeLevelTopN(level_topn_conf_str)) {
    LOG_ERROR << "[" << section << "] "
                   << "parse " << kConfigTreeLevelTopN << " failed, "
                   << "conf_str:" << level_topn_conf_str;
    return false;
  }
  LOG_INFO << "[" << section << "] "
                << kConfigTreeLevelTopN << ": " << level_topn_conf_str;

  // item_feature_group_id
  if (!conf_parser.GetValue<std::string>(
      section, kConfigItemFeatureGroupId, &item_feature_group_id_) ||
      item_feature_group_id_.empty()) {
    LOG_ERROR << "[" << section << "] get config "
                   << kConfigItemFeatureGroupId << " failed";
    return false;
  }
  LOG_INFO << "[" << section << "] "
                << kConfigItemFeatureGroupId << ": " << item_feature_group_id_;

  return true;
}

bool TreeIndexConf::ParseTreeLevelTopN(const std::string& conf_str) {
  char buf[512];
  snprintf(buf, sizeof(buf), "%s", conf_str.c_str());

  std::vector<char*> split_1;
  std::vector<char*> split_2;

  util::StrUtil::Split(buf, ';', true, &split_1);
  for (size_t i = 0; i < split_1.size(); i++) {
    util::StrUtil::Split(split_1[i], ',', true, &split_2);
    if (split_2.size() != 2) {
      LOG_ERROR << "Parse conf failed, pair size != 2";
      return false;
    }
    uint32_t level;
    uint32_t topn;
    if (!util::StrUtil::StrConvert<uint32_t>(split_2[0], &level)) {
      LOG_ERROR << "Parse conf failed, level not number";
      return false;
    }
    if (!util::StrUtil::StrConvert<uint32_t>(split_2[1], &topn)) {
      LOG_ERROR << "Parse conf failed, topn not number";
      return false;
    }
    if (topn == 0) {
      LOG_ERROR << "Parse conf failed, topn is 0";
      return false;
    }
    level_to_topn_[level] = topn;
  }

  return true;
}

uint32_t TreeIndexConf::tree_level_topn(uint32_t level) const {
  std::tr1::unordered_map<uint32_t, uint32_t>::const_iterator iter
      = level_to_topn_.find(level);
  if (iter != level_to_topn_.end()) {
    return iter->second;
  } else {
    return kDefaultTreeLevelTopN;
  }
}

}  // namespace tdm_serving
