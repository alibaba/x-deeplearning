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

#include "util/conf_parser.h"
#include <errno.h>
#include <fstream>
#include "util/str_util.h"
#include "util/log.h"

namespace tdm_serving {
namespace util {

const char* SECTION_START_TAG = "[";
const char* SECTION_END_TAG = "]";
const char* KVPAIR_TAG = "=";

const char* REGEX_PATTERN_SECTION = "^\\[(.*)\\]$";
const char* REGEX_PATTERN_KVPAIR = "(.*?)\\=(.*)";

// conf kv impl
ConfKv::ConfKv(const std::string& key, const std::string& value)
    : key_(new std::string(key)), value_(new std::string(value)) {
}

ConfKv::~ConfKv() {
  Clear();
}

void ConfKv::Clear() {
  DELETE_AND_SET_NULL(key_);
  DELETE_AND_SET_NULL(value_);
}

const std::string& ConfKv::GetKey() const {
  return *key_;
}

const std::string& ConfKv::GetValue() const {
  return *value_;
}

// conf section impl
ConfSection::ConfSection(const std::string& section)
    : section_(section) {
}

ConfSection::~ConfSection() {
  Clear();
}

void ConfSection::Clear() {
  for (uint32_t i = 0; i < kv_vec_.size(); ++i) {
    DELETE_AND_SET_NULL(kv_vec_[i]);
  }
  kv_vec_.clear();
  kv_map_.clear();
}

const std::string& ConfSection::GetSectionName() const {
  return section_;
}

const std::vector<ConfKv*>& ConfSection::GetAllKvs() const {
  return kv_vec_;
}

const ConfKv* ConfSection::GetKv(const std::string& key) const {
  std::tr1::unordered_map<std::string, ConfKv*>::const_iterator iter;
  iter = kv_map_.find(key);
  if (iter == kv_map_.end()) {
    return NULL;
  }
  return iter->second;
}

bool ConfSection::AddKv(const std::string& key, const std::string& value) {
  if (GetKv(key) != NULL) {
    LOG_ERROR << "section [" << section_ << "] "
                   << "has duplicate key [" << key << "]";
    return false;
  }
  ConfKv* new_kv = new ConfKv(key, value);
  kv_map_[key] = new_kv;
  kv_vec_.push_back(new_kv);
  return true;
}

// conf parser impl
ConfParser::ConfParser() {
}

ConfParser::~ConfParser() {
  Clear();
}

void ConfParser::Clear() {
  for (uint32_t i = 0; i < conf_section_vec_.size(); ++i) {
    DELETE_AND_SET_NULL(conf_section_vec_[i]);
  }
  conf_section_vec_.clear();
  conf_section_map_.clear();
}

bool ConfParser::Init(const std::string& kv_conf) {
  // clear
  Clear();
  // open and parse
  std::ifstream file(kv_conf.c_str());
  if (!file) {
    LOG_ERROR << "open " << kv_conf << " failed";
    return false;
  }

  std::string line = "";
  std::string section = "";
  std::string key = "";
  std::string value = "";
  size_t kv_tag_pos = 0;

  ConfSection* conf_section = NULL;
  while (std::getline(file, line)) {
    StrUtil::Trim(line);
    if (line.empty() || line.find("//") == 0 || line.find("#") == 0) {
      continue;
    }

    if (StrUtil::SubStr(line, SECTION_START_TAG, SECTION_END_TAG, &section)) {
      if (GetConfSection(section) != NULL) {
        LOG_ERROR << "find duplicate section [" << section << "]";
        return false;
      }
      conf_section = AddConfSection(section);
    } else if ((kv_tag_pos = line.find(KVPAIR_TAG)) != std::string::npos) {
      key = line.substr(0, kv_tag_pos);
      StrUtil::Trim(key);
      value = line.substr(kv_tag_pos + 1);
      StrUtil::Trim(value);
      if (key.empty(), value.empty()) {
        LOG_WARN << "skip empty key or value item for key";
        continue;
      }
      if (!conf_section->AddKv(key, value)) {
        LOG_ERROR <<
            "inser key [" << key << "] value [" << value << "] failed";
        return false;
      }
    }
  }

  return true;
}

const std::vector<ConfSection*>& ConfParser::GetAllConfSection() const {
  return conf_section_vec_;
}

const ConfSection*
ConfParser::GetConfSection(const std::string& section) const {
  std::tr1::unordered_map<std::string, ConfSection*>::const_iterator iter;
  iter = conf_section_map_.find(section);
  if (iter == conf_section_map_.end()) {
    return NULL;
  } else {
    return iter->second;
  }
}

ConfSection* ConfParser::AddConfSection(const std::string& section) {
  ConfSection* conf_section = const_cast<ConfSection*>(GetConfSection(section));
  if (conf_section == NULL) {
    conf_section = new ConfSection(section);
    conf_section_vec_.push_back(conf_section);
    conf_section_map_[section] = conf_section;
  }
  return conf_section;
}

}  // namespace util
}  // namespace tdm_serving

